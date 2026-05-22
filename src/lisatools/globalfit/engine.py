"""Engine and configuration dataclasses for the LISA global-fit sampler."""

from __future__ import annotations

import dataclasses
import logging
from collections import namedtuple
import os
from typing import Any, Callable, Optional, Union

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
from ..domains import (
    DomainSettingsBase,
    FDSettings,
    STFTSettings,
    WDMLookupTable,
    WDMSettings,
)
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
    """MPI rank assignments for the global fit.

    Args:
        head_rank: Rank that orchestrates the run.
        main_rank: Rank that drives the main sampler loop.
    """

    head_rank: int = -1
    main_rank: int = -1


class Setup:
    """Adapter wrapping a :class:`Settings` dataclass for use as a setup object.

    Args:
        settings_holder: A :class:`Settings` instance whose fields populate
            this :class:`Setup`. The dataclass field metadata is introspected
            so the same constructor can be re-applied.
    """

    def __init__(self, settings_holder: Settings):
        self._settings_class = type(settings_holder)
        self.settings = settings_holder

        # had a better way to do this but it stopped allowing for pickle
        self._settings_names = [field.name for field in dataclasses.fields(self._settings_class)]
        self._settings_args_names = [
            field.name
            for field in dataclasses.fields(self._settings_class)
            if (
                field.default == dataclasses.MISSING
                and field.default_factory == dataclasses.MISSING
            )
        ]  # args
        self._settings_kwargs_names = [
            field.name
            for field in dataclasses.fields(self._settings_class)
            if (
                field.default != dataclasses.MISSING or field.default_factory != dataclasses.MISSING
            )
        ]  # kwargs
        _args = tuple([getattr(settings_holder, key) for key in self._settings_args_names])
        _kwargs = {key: getattr(settings_holder, key) for key in self._settings_kwargs_names}
        self._settings_class.__init__(self, *_args, **_kwargs)
        self.init_df()

    @property
    def settings(self) -> Settings:
        """The wrapped :class:`Settings` object."""
        return self._settings

    @settings.setter
    def settings(self, settings: Settings):
        assert isinstance(settings, Settings)
        self._settings = settings

    def init_df(self):
        """Round ``Tobs`` to a multiple of ``dt`` and recompute ``df = 1/Tobs``."""
        self.Tobs = int(self.Tobs / self.dt) * self.dt
        self.df = 1.0 / self.Tobs


@dataclasses.dataclass
class Settings:
    """Common configuration fields shared by branch-specific setup objects.

    Most fields default to ``None`` so subclasses can extend or override
    them; users assemble settings either by subclassing or by setting fields
    directly.
    """
    Tobs: float | None = None
    dt: float | None = None
    initialize_kwargs: dict | None = None
    transform: Optional[TransformContainer] = None
    priors: Optional[typing.Dict[str,ProbDistContainer]] = None
    periodic: Optional[dict] = None
    nleaves_max: Optional[int] = None
    nleaves_min: Optional[int] = None
    ndim: Optional[int] = None
    betas: Optional[np.ndarray] = None
    other_tempering_kwargs: Optional[dict] = None
    branch_state: Optional[eryn_State] = None
    branch_backend: Optional[eryn_Backend] = None
    log_dir: Optional[str] = None


# Type alias: a user-supplied domain spec is either a fully constructed
# DomainSettings instance (used as-is) or a factory of signature
# ``(times: np.ndarray, dt: float, force_backend: str) -> DomainSettingsBase``
# which the engine calls *after* loading the data so the grid can be sized
# against ``times`` / ``dt`` if needed. The factory form is the only way to
# pass an STFT grid (which depends on ``times``); WDM and FD typically pass
# a pre-constructed instance.
DomainSettingsSpec = Union[DomainSettingsBase, Callable[..., DomainSettingsBase]]


@dataclasses.dataclass
class GeneralSettings(Settings):
    """Top-level (non-source-specific) configuration for a global-fit run.

    Holds output paths, observation/window options, the basis-domain choice
    (passed as a :class:`DomainSettingsBase` instance or a factory callable),
    and the ``data_processor`` that loads/conditions the input data. Field
    defaults of ``None`` mark options that the user must supply (see
    :meth:`GeneralSetup.init_setup` for assertions).
    """
    Tobs: float | None = None
    dt: float | None = None
    file_store_dir: str | None = None
    base_file_name: str | None = None
    main_file_key: Optional[str] = "parameter_estimation_main"
    past_file_for_start: Optional[str] = None
    orbits: Orbits | None = None
    gpu_orbits: Orbits | None = None
    # Pass either a constructed DomainSettings instance or a factory
    # callable ``(times, dt, force_backend) -> DomainSettingsBase``. The
    # callable form is used by the STFT path where the grid depends on the
    # loaded ``times`` array; FD/WDM users typically construct the settings
    # directly. No string-level domain flag.
    domain_settings: Optional[DomainSettingsSpec] = None
    # Optional WDM lookup table. When the basis settings resolve to a
    # WDMSettings, the user supplies the pre-built lookup table here (or
    # passes a factory ``(WDMSettings) -> WDMLookupTable``). Stored on the
    # setup so downstream GB code can pick it up. Ignored otherwise.
    wdm_lookup_table: Optional[
        Union[WDMLookupTable, Callable[[WDMSettings], WDMLookupTable]]
    ] = None
    random_seed: int | None = None
    backup_iter: int | None = None
    nwalkers: int | None = None
    ntemps: int | None = None
    window_type: str = "tukey"
    window_taper_duration: float | None = None
    gpu_backend: str = "cuda12x"
    gpus: typing.List[int] | None = None
    fixed_psd_kwargs: typing.Dict[str, typing.Any] | None = None
    data_processor: Optional[BaseProcessingStep] = None
    processor_init_kwargs: Optional[dict] = None
    preprocess_kwargs: Optional[dict] = None
    sensitivity_init_kwargs: Optional[dict] = None
    normalize_window: bool = False
    catalogue: typing.Optional[dict] = None


from .loginfo import init_logger


class GeneralSetup(Setup, GeneralSettings):
    """Setup object that ingests data, prepares orbits, and builds the sensitivity backend.

    On construction it:

    1. Ensures the artifacts directory exists and attaches a logger.
    2. Runs the configured :attr:`data_processor` to load and condition the
       time-domain data.
    3. Resolves ``domain_settings`` (instance or factory) into a concrete
       :class:`~lisatools.domains.DomainSettingsBase`, builds the analysis
       window, and constructs the input :class:`DataResidualArray`.
    4. Resolves ``wdm_lookup_table`` (instance or factory) when running in
       the WDM domain.
    5. Configures :class:`XYZSensitivityBackend` for use in PSD/likelihood
       calls.

    Args:
        general_settings: The :class:`GeneralSettings` to use.
    """

    def __init__(self, general_settings: GeneralSettings):

        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, general_settings)

        level = logging.DEBUG
        name = "GeneralSetup"
        # Ensure artifacts dir exists before creating log file handler
        if not os.path.exists(self.artifacts_file_dir):
            os.makedirs(self.artifacts_file_dir)
        self.logger = init_logger(filename="general_setup.log", level=level, name=name, log_dir=self.artifacts_file_dir)

        self.init_setup()

    @property
    def main_file_path(self) -> str:
        """Filesystem path of the primary HDF5 backend file."""
        return self.file_store_dir + self.base_file_name + "_" + self.main_file_key + ".h5"

    @property
    def artifacts_file_dir(self) -> str:
        """Directory where logs and diagnostic plots are stored for this run."""
        return self.file_store_dir + self.base_file_name + "_artifacts/"

    def init_setup(self):
        """Validate required settings and trigger data preparation."""
        if self.file_store_dir is None:
            raise ValueError("Must provide file_store_dir settings for GeneralSetup.")
        if self.base_file_name is None:
            raise ValueError("Must provide base_file_name settings for GeneralSetup.")
        if self.domain_settings is None:
            raise ValueError(
                "Must provide domain_settings (a DomainSettings instance or a "
                "factory ``(times, dt, force_backend) -> DomainSettingsBase``)."
            )

        # CPU fallback: when gpus is None, force_backend resolves to "cpu" so
        # downstream consumers stay numpy-only. GPU path keeps cupy.
        self.force_backend = self.gpu_backend if self.gpus is not None else "cpu"
        self.logger.debug(f"Saving h5 backend to {self.main_file_path}")
        self.logger.debug(f"Saving artifacts to {self.artifacts_file_dir}")
        if not os.path.exists(self.artifacts_file_dir):
            os.makedirs(self.artifacts_file_dir)
            self.logger.debug(f"Created artifacts directory")

        self.init_data_information()

    def init_orbit_information(self):
        """Construct orbit objects, defaulting to :class:`EqualArmlengthOrbits`."""
        if self.orbits is None:
            self.orbits = EqualArmlengthOrbits()
            self.gpu_orbits = EqualArmlengthOrbits(force_backend=self.force_backend)
        else:
            if self.gpu_orbits is None and self.force_backend == self.gpu_backend:
                # TODO: make better
                raise ValueError("If adding orbits, make sure to duplicate into GPU orbits.")

    def _resolve_domain_settings(
        self, times: np.ndarray, dt: float
    ) -> DomainSettingsBase:
        """Turn the user-supplied ``domain_settings`` into a concrete instance.

        Accepts either a :class:`DomainSettingsBase` (used directly) or a
        factory called with ``(times, dt, force_backend)``.
        """
        spec = self.domain_settings
        if isinstance(spec, DomainSettingsBase):
            return spec
        if callable(spec):
            return spec(times=times, dt=dt, force_backend=self.force_backend)
        raise TypeError(
            f"domain_settings must be a DomainSettingsBase instance or a "
            f"factory callable; got {type(spec).__name__}."
        )

    def _resolve_wdm_lookup_table(
        self, domain_settings: WDMSettings
    ) -> Optional[WDMLookupTable]:
        """Resolve the (optional) WDM lookup table from the user-supplied spec.

        ``wdm_lookup_table`` may be a fully constructed :class:`WDMLookupTable`
        (used directly), a factory ``(WDMSettings) -> WDMLookupTable``
        (called now that the grid is known), or ``None``.
        """
        spec = self.wdm_lookup_table
        if spec is None:
            return None
        if isinstance(spec, WDMLookupTable):
            return spec
        if callable(spec):
            return spec(domain_settings)
        raise TypeError(
            f"wdm_lookup_table must be a WDMLookupTable instance or a factory; "
            f"got {type(spec).__name__}."
        )

    def init_data_information(self):
        """Run preprocessing, build the basis domain, and configure the sensitivity backend."""
        if self.data_processor is None:
            raise ValueError("Must provide data_processor for GeneralSetup.")

        data_processor = self.data_processor(**(self.processor_init_kwargs or {}))

        # preprocess data
        if self.fixed_psd_kwargs is None:

            self.fixed_psd_kwargs = dict(
                psd_params=[15e-12, 3e-15],  # default scirdv1
                galfor_params=None,
            )

        default_preprocess_kwargs = dict(
            plot_folder=self.artifacts_file_dir,
            highpass_kwargs=dict(cutoff=2e-5, order=2, zero_phase=True),
            trim_kwargs=dict(duration=200 * 3600, is_percent=False, trimming_type="from_each_end"),
            Tobs=self.Tobs,
        )

        if self.preprocess_kwargs is None:
            preprocess_kwargs = default_preprocess_kwargs
        else:
            preprocess_kwargs = {**default_preprocess_kwargs, **self.preprocess_kwargs}

        for key, value in preprocess_kwargs.items():
            self.logger.debug(f"Preprocess setting: {key} = {value}")

        times, _ = data_processor.process(**preprocess_kwargs)
        dt = data_processor.td_signal.settings.dt
        Nt = len(times)
        self.data_t0 = float(times[0])
        self.catalogue = getattr(data_processor, 'catalogue', {})

        domain_settings = self._resolve_domain_settings(times=times, dt=dt)
        self.basis_kwargs = dict(force_backend=self.force_backend)

        # Domain-specific window length + diagnostic-plot kwargs. The window
        # is always built on the underlying time grid (length Nt); the basis
        # transform inside data_processor.pour() handles the projection.
        if isinstance(domain_settings, STFTSettings):
            nperseg = domain_settings.get_nperseg(dt)
            self.window_alpha = self.window_taper_duration / (nperseg * dt)
            window, _ = windowfun(self.window_type, nperseg, alpha=self.window_alpha)
            plot_kwargs_list = [
                dict(channel=0, plot_type="stft", filename=self.artifacts_file_dir + "stft_data.png"),
                dict(channel=0, plot_type="fd", time_bin=0, filename=self.artifacts_file_dir + "fd_data.png"),
                dict(channel=0, plot_type="td", freq_bin=0, filename=self.artifacts_file_dir + "td_data.png"),
            ]
        elif isinstance(domain_settings, FDSettings):
            self.window_alpha = self.window_taper_duration / (Nt * dt)
            window, _ = windowfun(self.window_type, Nt, alpha=self.window_alpha)
            plot_kwargs_list = [
                dict(channel=0, filename=self.artifacts_file_dir + "fd_data.png")
            ]
        elif isinstance(domain_settings, WDMSettings):
            self.window_alpha = self.window_taper_duration / (Nt * dt)
            window, _ = windowfun(self.window_type, Nt, alpha=self.window_alpha)
            # WDMSignal exposes a heatmap rather than a generic plot; the
            # engine renders it directly after pouring the data in.
            plot_kwargs_list = []
        else:
            raise NotImplementedError(
                f"Basis domain {type(domain_settings).__name__} not implemented."
            )

        self.domain_settings = domain_settings
        self.input_data_residual_array, orbits = data_processor.pour(
            settings=domain_settings, window=window, return_orbits=True
        )

        if isinstance(domain_settings, FDSettings):
            # FD path: thread the active-band f_arr through DataResidualArray
            # so downstream consumers (e.g. moves keyed off start_freq_ind)
            # see the correct band.
            self.input_data_residual_array.data_length = len(domain_settings.f_arr)
            self.input_data_residual_array._store_time_and_frequency_information(
                df=domain_settings.df,
                f_arr=domain_settings.f_arr,
            )

        # Diagnostic plots — domain-dependent.
        for plot_kwargs_here in plot_kwargs_list:
            _ = self.input_data_residual_array.data_res_arr.plot(**plot_kwargs_here)

        if isinstance(domain_settings, WDMSettings):
            # WDMSignal.heatmap() takes no `filename` kwarg; render and save
            # manually so we still get a diagnostic figure.
            import matplotlib.pyplot as plt

            try:
                fig, _ = self.input_data_residual_array.data_res_arr.heatmap(mag=True)
                fig.savefig(self.artifacts_file_dir + "wdm_data.png", bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                # Heatmap is purely diagnostic; never fail the run on plot errors.
                self.logger.warning(f"WDM data heatmap failed: {exc}")

        # log domain info
        for key, value in domain_settings.__dict__.items():
            self.logger.info(f"Domain setting: {key} = {value}")

        if orbits is not None:
            self.orbits = orbits
            if self.force_backend == self.gpu_backend:
                self.gpu_orbits = data_processor.orbits_class(
                    filename=orbits.filename,
                    armlength=orbits.armlength,
                    force_backend=self.gpu_backend,
                )

        self.init_orbit_information()

        # Resolve the (optional) WDM lookup table once the WDM grid is final.
        if isinstance(domain_settings, WDMSettings):
            self.wdm_lookup_table = self._resolve_wdm_lookup_table(domain_settings)
        else:
            # Forbid a stray lookup table on FD/STFT runs — it would never be
            # used and signals a setup mistake.
            if self.wdm_lookup_table is not None:
                raise ValueError(
                    "wdm_lookup_table was provided but domain_settings is not "
                    "a WDMSettings; remove it or switch the domain."
                )

        # TODO: AET sensitivity backend wiring.
        self.sensitivity_backend = XYZSensitivityBackend(
            orbits=self.gpu_orbits,
            settings=domain_settings,
            force_backend=self.force_backend,
            window_values=window if self.normalize_window else None,
            **self.sensitivity_init_kwargs,
        )


@dataclasses.dataclass
class GlobalFitSettings:
    """Top-level settings bundle describing one global-fit run.

    Args:
        source_info: Mapping of source-class name to its :class:`Setup`.
        general_info: General (non-source) settings.
        rank_info: MPI rank assignments.
        setup_function: User-supplied callback invoked once at run start to
            wire the global fit's components together.
    """

    source_info: typing.Dict[str, Setup]
    general_info: GeneralSetup
    rank_info: RankInfo
    # TODO: add to adocs current args for these
    setup_function: typing.Callable[(...), None]


@dataclasses.dataclass
class EngineInfo:
    """Branch-keyed metadata that the engine needs in order to launch a sampler."""

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
    """``eryn`` :class:`EnsembleSampler` extended with a shared analysis-container array.

    The engine owns the :class:`AnalysisContainerArray` shared by all moves so
    likelihoods, residuals, and PSDs stay synchronized across walker indices
    during the run.

    Args:
        analysis_container_arr: Shared :class:`AnalysisContainerArray` for
            this run.
        *args: Forwarded to :class:`eryn.ensemble.EnsembleSampler`.
        **kwargs: Forwarded to :class:`eryn.ensemble.EnsembleSampler`.
    """

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
