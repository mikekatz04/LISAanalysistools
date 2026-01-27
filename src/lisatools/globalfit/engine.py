from __future__ import annotations
from os import times
from eryn.ensemble import EnsembleSampler
from collections import namedtuple
from typing import Optional
import dataclasses
import logging
import h5py
import numpy as np


__all__ = ["GlobalFitInfo", "GlobalFitEngine"]


import typing 

from ..utils.utility import tukey, detrend, AET

from eryn.backends import backend as eryn_Backend
from eryn.state import State as eryn_State
from eryn.utils.transform import TransformContainer
from eryn.prior import ProbDistContainer


from .utils import NewSensitivityMatrix
from lisatools.detector import Orbits, EqualArmlengthOrbits
from ..detector import sangria, mojito, LISAModel
from ..sensitivity import XYZ1SensitivityMatrix, XYZ2SensitivityMatrix, AE1SensitivityMatrix, AE2SensitivityMatrix, AET2SensitivityMatrix
from ..mojito_detector import XYZSensitivityBackend
from .preprocessing import BaseProcessingStep


@dataclasses.dataclass
class RankInfo:
    head_rank: int = -1
    main_rank: int = -1


class Setup:

    def __init__(self, settings_holder: Settings):
        self._settings_class = type(settings_holder)
        self.settings = settings_holder

        # had a better way to do this but it stopped allowing for pickle
        self._settings_names = [field.name for field in dataclasses.fields(self._settings_class)]
        self._settings_args_names = [field.name for field in dataclasses.fields(self._settings_class) if (field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING)]  # args
        self._settings_kwargs_names = [field.name for field in dataclasses.fields(self._settings_class) if (field.default != dataclasses.MISSING or field.default_factory != dataclasses.MISSING)]  # kwargs
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
        self.df = 1. / self.Tobs
    

@dataclasses.dataclass
class Settings:
    Tobs: float = None
    dt: float = None
    initialize_kwargs: dict = None
    transform: Optional[TransformContainer] = None
    priors: Optional[ProbDistContainer] = None
    periodic: Optional[dict] = None
    nleaves_max: Optional[int] = None # TODO: need to make higher
    nleaves_min: Optional[int] = None
    ndim: Optional[int] = None
    betas: Optional[np.ndarray] = None
    other_tempering_kwargs: Optional[dict] = None
    branch_state: Optional[eryn_State] = None
    branch_backend: Optional[eryn_Backend] = None


@dataclasses.dataclass
class GeneralSettings(Settings):
    Tobs: float = None
    dt: float = None
    file_store_dir: str = None
    base_file_name: str = None
    main_file_key: Optional[str] = "parameter_estimation_main"
    past_file_for_start: Optional[str] = None
    orbits: Orbits = None
    gpu_orbits: Orbits = None
    basis_domain: str = "stft"
    start_freq: float = None
    end_freq: float = None
    stft_dt: float = None
    random_seed: int = None
    backup_iter: int = None
    nwalkers: int = None
    ntemps: int = None
    tukey_alpha: float = None
    gpus: typing.List[int] = None
    fixed_psd_kwargs: typing.Dict[str, typing.Any] = None
    #channels: typing.List[str] = dataclasses.field(default_factory=lambda: ["A", "E"])
    #noise_model: Optional[LISAModel] = None
    data_processor: Optional[BaseProcessingStep] = None
    processor_init_kwargs: Optional[dict] = None
    preprocess_kwargs: Optional[dict] = None
    sensitivity_init_kwargs: Optional[dict] = None
    # file_information["gb_main_chain_file"] = file_store_dir + base_file_name + "_gb_main_chain_file.h5"
    # file_information["gb_all_chain_file"] = file_store_dir + base_file_name + "_gb_all_chain_file.h5"

    # file_information["mbh_main_chain_file"] = file_store_dir + base_file_name + "_mbh_main_chain_file.h5"
    # file_information["mbh_search_file"] = file_store_dir + base_file_name + "_mbh_search_tmp_file.h5"
    
    
from .loginfo import init_logger

class GeneralSetup(Setup, GeneralSettings):
    def __init__(self, general_settings: GeneralSettings):
        
        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, general_settings)

        level = logging.DEBUG
        name = "GeneralSetup"
        self.logger = init_logger(filename="gb_setup.log", level=level, name=name)
        
        self.init_setup()

    @property
    def data_length(self) -> int:
        return len(self.A_inj)
    
    @property
    def main_file_path(self) -> str:
        return (
            self.file_store_dir 
            + self.base_file_name 
            + "_" + self.main_file_key
            + ".h5"
        )

    # def __getattr__(self, attr: str) -> typing.Any:
    #     if hasattr(self.gb_settings, attr):
    #         return getattr(self.gb_settings, attr)

    def init_setup(self):
        if self.file_store_dir is None:
            raise ValueError("Must provide file_store_dir settings for GeneralSetup.")
        if self.base_file_name is None:
            raise ValueError("Must provide base_file_name settings for GeneralSetup.")
        # if self.data_input_path is None:    
        #     raise ValueError("Must provide base_file_name settings for GeneralSetup.")

        self.init_data_information()

    def init_orbit_information(self):
        if self.orbits is None:
            self.orbits = EqualArmlengthOrbits()
            self.gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")
        else:
            if self.gpu_orbits is None:
                # TODO: make better
                raise ValueError("If adding orbits, make sure to duplicate into GPU orbits.")

    def init_data_information(self):

        #load data #todo add here not in file
        if self.data_processor is None:
            raise ValueError("Must provide data_processor for GeneralSetup.")
        
        data_processor = self.data_processor(**(self.processor_init_kwargs or {}))

        # preprocess data
        if self.fixed_psd_kwargs is None:
            
            self.fixed_psd_kwargs = dict(
                psd_params = [15e-12, 3e-15],  # default scirdv1
                galfor_params = None,
            )
        
        default_preprocess_kwargs = dict(
            do_detrend = True,
            highpass_kwargs = dict(cutoff=5e-6, order=2, zero_phase=True),
            trim_kwargs = dict(duration=200 * 3600, is_percent=False, trimming_type='from_each_end'),
            Tobs = self.Tobs,
        )

        if self.preprocess_kwargs is None:
            preprocess_kwargs = default_preprocess_kwargs
        else:
            preprocess_kwargs = {**default_preprocess_kwargs, **self.preprocess_kwargs}

        times, _ = data_processor.process(**preprocess_kwargs)
        dt = data_processor.td_signal.settings.dt
        Nt = len(times)

        if self.basis_domain == "stft":
            from ..domains import get_stft_settings

            if self.stft_dt is None:
                raise ValueError("Must provide `stft_dt` for stft basis domain.")
            domain_settings = get_stft_settings(times=times, big_dt=self.stft_dt, min_freq=self.start_freq, max_freq=self.end_freq)
            nperseg = domain_settings.get_nperseg(dt)
            window = tukey(nperseg, alpha=self.tukey_alpha)
        
        elif self.basis_domain == "fd":
            from ..domains import FDSettings

            df = 1. / (Nt * dt)
            Nf = Nt // 2 + 1
            domain_settings = FDSettings(N=Nf, df=df, min_freq=self.start_freq, max_freq=self.end_freq)
            window = tukey(Nt, alpha=self.tukey_alpha)
        
        else:
            raise NotImplementedError(f"Basis domain {self.basis_domain} not implemented.")
        
        self.input_data_residual_array, orbits = data_processor.pour(settings = domain_settings, window=window, return_orbits=True)
        
        # use logger to output domain info

        for key, value in domain_settings.__dict__.items():
            self.logger.info(f"Domain setting: {key} = {value}")

        if orbits is not None:
            self.orbits = orbits
            self.gpu_orbits = data_processor.orbits_class(filename=orbits.filename, armlength=orbits.armlength, force_backend="cuda12x")
            #self.gpu_orbits.configure()
        
        self.init_orbit_information()

        #todo make it flexible when adding also AET backend.
        self.sensitivity_backend = XYZSensitivityBackend(orbits=self.gpu_orbits,
                                                        settings=domain_settings,
                                                        **self.sensitivity_init_kwargs,
                                                        )
        
        # if self.noise_model.name == 'sangria':
        #     if "A" in self.channels:
        #         sens_fns = [
        #             'A1TDISens',
        #             'E1TDISens',
        #         ]
        #         self.sensitivity_matrix = AE1SensitivityMatrix
        #     elif "X" in self.channels:
        #         sens_fns = XYZ1SensitivityMatrix
        #         self.sensitivity_matrix = XYZ1SensitivityMatrix

        # elif self.noise_model.name == 'mojito':
            
        #     if "A" in self.channels:
        #         sens_fns = [
        #             'A2TDISens',
        #             'E2TDISens',
        #             'T2TDISens',
        #         ][:len(self.channels)]
        #         self.sensitivity_matrix = AET2SensitivityMatrix if len(self.channels) == 3 else AE2SensitivityMatrix
        #     else:
                
        #         sens_fns = XYZ2SensitivityMatrix
        #         self.sensitivity_matrix = XYZ2SensitivityMatrix

        # self.new_sens_mat = NewSensitivityMatrix(
        #         orbits=self.orbits,
        #         noise_model=self.noise_model,
        #         sens_fns=sens_fns,
        # )



@dataclasses.dataclass
class GlobalFitSettings:
    source_info: typing.Dict[str, Setup]
    general_info: GeneralSetup
    rank_info: RankInfo
    # TODO: add to adocs current args for these
    setup_function: typing.Callable[(...), None]


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
        """Get ``Model`` object from sampler

        The model object is used to pass necessary information to the
        proposals. This method can be used to retrieve the ``model`` used
        in the sampler from outside the sampler.

        Returns:
            :class:`Model`: ``Model`` object used by sampler.

        """
        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        # setup model framework for passing necessary items
        model = GlobalFitInfo(
            self.analysis_container_arr,
            map_fn,
            self._random,
        )
        return model
