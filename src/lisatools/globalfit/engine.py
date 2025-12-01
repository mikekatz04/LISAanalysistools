from __future__ import annotations
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



from lisatools.detector import Orbits


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
    data_input_path: str = None
    main_file_key: Optional[str] = "parameter_estimation_main"
    past_file_for_start: Optional[str] = None
    orbits: Orbits = None
    gpu_orbits: Orbits = None
    start_freq_ind: int = None
    end_freq_ind: int = None
    random_seed: int = None
    backup_iter: int = None
    nwalkers: int = None
    ntemps: int = None
    tukey_alpha: float = None
    gpus: typing.List[int] = None
    remove_from_data: typing.List[str] = None
    fixed_psd_kwargs: typing.Dict[str, typing.Any] = None

    # file_information["gb_main_chain_file"] = file_store_dir + base_file_name + "_gb_main_chain_file.h5"
    # file_information["gb_all_chain_file"] = file_store_dir + base_file_name + "_gb_all_chain_file.h5"

    # file_information["mbh_main_chain_file"] = file_store_dir + base_file_name + "_mbh_main_chain_file.h5"
    # file_information["mbh_search_file"] = file_store_dir + base_file_name + "_mbh_search_tmp_file.h5"
    
    
from ..detector import sangria
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
        if self.data_input_path is None:    
            raise ValueError("Must provide base_file_name settings for GeneralSetup.")

        self.init_data_information()
        self.init_orbit_information()

    def init_orbit_information(self):
        if self.orbits is None:
            self.orbits = EqualArmlengthOrbits()
            self.gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")
        else:
            if self.gpu_orbits is None:
                # TODO: make better
                raise ValueError("If adding orbits, make sure to duplicate into GPU orbits.")

    def init_data_information(self):

        if self.fixed_psd_kwargs is None:
            self.fixed_psd_kwargs = dict(
                model=sangria, 
            )
        if self.remove_from_data is None:
            self.remove_from_data = []

        assert isinstance(self.remove_from_data, list)

        # TODO: Generalize input. 
        with h5py.File(self.data_input_path, "r") as f:
            if "noise" not in self.remove_from_data:
                tXYZ = f["obs"]["tdi"][:]

                # remove sources
                for source in self.remove_from_data:  # , "dgb", "igb"]:  # "vgb" ,
                    if source == "noise":
                        continue
                    print(f"Removing {source} from data injection.")
                    change_arr = f["sky"][source]["tdi"][:]
                    for change in ["X", "Y", "Z"]:
                        tXYZ[change] -= change_arr[change]

            else: 
                keys = list(f["sky"])
                print("Initial keys in data injection: ", keys) 
                tmp_keys = keys.copy()
                for key in tmp_keys:
                    print(key)
                    if key in self.remove_from_data:
                        keys.remove(key)
                        print(f"Removing {key} from data injection.")
                tXYZ = f["sky"][keys[0]]["tdi"][:]
                for key in keys[1:]:
                    tXYZ["X"] += f["sky"][key]["tdi"][:]["X"]
                    tXYZ["Y"] += f["sky"][key]["tdi"][:]["Y"]
                    tXYZ["Y"] += f["sky"][key]["tdi"][:]["Z"]

        self.t, self.X, self.Y, self.Z = (
            tXYZ["t"].squeeze(),
            tXYZ["X"].squeeze(),
            tXYZ["Y"].squeeze(),
            tXYZ["Z"].squeeze(),
        )

        self.dt = self.t[1] - self.t[0]
        _Tobs = self.Tobs
        Nobs = int(_Tobs / self.dt)  # len(t)
        self.t = self.t[:Nobs]
        self.X = self.X[:Nobs]
        self.Y = self.Y[:Nobs]
        self.Z = self.Z[:Nobs]

        self.Tobs = Nobs * self.dt
        self.df = 1 / self.Tobs

        # TODO: @nikos what do you think about the window needed here. For this case at 1 year, I do not think it matters. But for other stuff.
        # the time domain waveforms like emris right now will apply this as well
        tukey_here = tukey(self.X.shape[0], self.tukey_alpha)
        X = detrend(self.t, tukey_here * self.X.copy())
        Y = detrend(self.t, tukey_here * self.Y.copy())
        Z = detrend(self.t, tukey_here * self.Z.copy())

        import matplotlib.pyplot as plt
        from emritools.plotting.waveforms import plot_sft

        _ = plot_sft(X, 24*3600, self.dt, fmin=1e-3); plt.yscale('log'); plt.savefig(f'{self.file_store_dir}/Xchannel_sft.png'); plt.close()

        # f***ing dt
        Xf, Yf, Zf = (np.fft.rfft(X) * self.dt, np.fft.rfft(Y) * self.dt, np.fft.rfft(Z) * self.dt)
        Af, Ef, Tf = AET(Xf, Yf, Zf)
        # Af[:] = 0.0
        # Ef[:] = 0.0
        # Tf[:] = 0.0

        if self.start_freq_ind is None:
            self.start_freq_ind = 0
        if self.end_freq_ind is None:
            self.end_freq_ind = len(Af) + self.start_freq_ind


        self.A_inj, self.E_inj = (
            Af[self.start_freq_ind:self.end_freq_ind],
            Ef[self.start_freq_ind:self.end_freq_ind],
        )

        self.fd = (np.arange(len(self.A_inj)) + self.start_freq_ind) * self.df

        
        # TODO: clean this up
        assert len(self.t) == len(self.X) == len(self.Y) == len(self.Z)
        assert len(self.fd) == len(self.A_inj) == len(self.E_inj)

        assert len(self.A_inj) == self.end_freq_ind - self.start_freq_ind


@dataclasses.dataclass
class GlobalFitSettings:
    source_info: typing.Dict[str, Setup]
    general_info: GeneralSetup
    rank_info: RankInfo
    # TODO: add to adocs current args for these
    setup_function: typing.Callable[(...), None]


import dataclasses
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
