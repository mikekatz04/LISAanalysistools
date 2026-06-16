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
from ..utils.utility import AET, detrend, windowfun, generate_multivariate_noise_fd, interpolate_complex_noise
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
    num_iterations: int | None = 500
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
    gpu_backend: str = "cuda13x"
    gpus: typing.List[int] | None = None
    fixed_psd_kwargs: typing.Dict[str, typing.Any] | None = None
    # channels: typing.List[str] = dataclasses.field(default_factory=lambda: ["A", "E"])
    # noise_model: Optional[LISAModel] = None
    data_processor: Optional[BaseProcessingStep] = None
    processor_init_kwargs: Optional[dict] = None
    preprocess_kwargs: Optional[dict] = None
    sensitivity_init_kwargs: Optional[dict] = None
    normalize_window: bool = False
    # catalogue: typing.Optional[dict] = None

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
    
    @property
    def catalogue(self):
        return getattr(self.data_processor, "catalogue", {})

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

        if hasattr(data_processor, "process"):
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
            
        else:
            assert self.basis_domain == "fd", "Pre-generated data only supported for fd basis domain currently."
            self.input_data_residual_array, orbits = data_processor.generate_data()
            self.orbits = data_processor.orbits
            domain_settings =  self.input_data_residual_array.settings
            assert self.dt is not None
            self.Tobs = 1.0 / domain_settings.df
            self.data_td_settings = TDSettings(
                t0=0.0, 
                N=int(self.Tobs / self.dt), 
                dt=self.dt, 
                force_backend=self.force_backend
            )
            plot_kwargs_list = [dict(channel=0, filename=self.artifacts_file_dir + "fd_data.png")]
            

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
        if self.sensitivity_init_kwargs is not None and "galactic_grid_kwargs" in self.sensitivity_init_kwargs:
            if "t0" not in self.sensitivity_init_kwargs["galactic_grid_kwargs"]:
                self.sensitivity_init_kwargs["galactic_grid_kwargs"]["t0"] = self.data_t0
                
        self.sensitivity_backend = XYZSensitivityBackend(
            orbits=self.gpu_orbits,
            settings=domain_settings,
            force_backend=self.force_backend,
            window_values=window if self.normalize_window else None,
            **self.sensitivity_init_kwargs,
        )
        
        if not hasattr(self.data_processor, "process"):
            # TODO: add option for empirical noise path in settings and load from there if provided
            self.empirical_noise_path = "/sps/lisaf/crondeel/mojito_light/data/INSTRUMENT/L1/NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5"
            self.logger.info(f"Loading empirical noise from {self.empirical_noise_path}")
            self._setup_empirical_sensitivity(domain_settings.f_arr)
            self.logger.info("Injecting multivariate correlated noise from SensitivityMatrix.")
            self._inject_correlated_noise(domain_settings.df)
        
            for plot_kwargs_here in plot_kwargs_list:
                plot_kwargs_list[0]["filename"] = self.artifacts_file_dir + "fd_data_with_noise.png"
                _ = self.input_data_residual_array.data_res_arr.plot(**plot_kwargs_here)
        
        # --- Store preprocessing --- #
        self.data_processor = data_processor
        
    def _setup_empirical_sensitivity(self, f_arr):
        with h5py.File(self.empirical_noise_path, "r") as f:
            # Assumes Mojito format: shape (N_times, N_freqs, 3, 3)
            CENTRAL_FREQ = 281600000000000.0
            xyz_noise_estimate = f['noise_estimates/XYZ'][:] / CENTRAL_FREQ**2
            
            fmin = f['noise_estimates/log_frequency_sampling'].attrs['fmin']
            fmax = f['noise_estimates/log_frequency_sampling'].attrs['fmax']
            size = f['noise_estimates/log_frequency_sampling'].attrs['size']
            noise_freqs = np.logspace(np.log10(fmin), np.log10(fmax), size)
            
        noise_estimate_avg = xyz_noise_estimate.mean(axis=0)

        interpolated_noise = interpolate_complex_noise(noise_freqs, noise_estimate_avg, f_arr)
        
        sens_mat_cpu = interpolated_noise.transpose(1, 2, 0)
        
        xp = self.sensitivity_backend.xp
        self.sensitivity_backend.sens_mat = xp.asarray(sens_mat_cpu)
        
        self.logger.debug("Empirical noise successfully pushed to SensitivityMatrixBase.")
    
    def _inject_correlated_noise(self, df):
        if self.empirical_noise_path is None:
            psd_params = self.fixed_psd_kwargs.get("psd_params", [15e-12, 3e-15])
            galfor_params = self.fixed_psd_kwargs.get("galfor_params", np.zeros(5))
            
            self.sensitivity_backend.set_sensitivity_matrix(
                Soms_d_in=psd_params[0], 
                Sa_a_in=psd_params[1],
                *(galfor_params if galfor_params is not None else np.zeros(5))
            )
        
        # Grab the currently active matrix (Empirical OR Analytic)
        sens_mat = self.sensitivity_backend.sens_mat 
        
        # Generate and inject
        noise_fd = generate_multivariate_noise_fd(sens_mat, df)
        self.input_data_residual_array.data_res_arr.arr += noise_fd
        
        self.logger.debug("Multivariate noise realization generated and added to signal.")
            
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
