"""Engine and configuration dataclasses for the LISA global-fit sampler."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
from collections import namedtuple
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
    TDSettings,
    WDMSettings,
)
from ..sensitivity import (
    AE1SensitivityMatrix,
    AE2SensitivityMatrix,
    AET2SensitivityMatrix,
    CompositeSensitivityBackend,
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
        head_rank: DEPRECATED legacy alias from the retired multi-stage
            pipeline; it has no role and should equal ``main_rank``. The
            dedicated saver rank is assigned automatically at ``np >= 3``
            (see ``GlobalFit.resolve_rank_roles``).
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
        """The wrapped :class:`Settings` object."""
        return self._settings

    @settings.setter
    def settings(self, settings: Settings):
        assert isinstance(settings, Settings)
        self._settings = settings

    @property
    def transform_fn(self) -> Optional[TransformContainer]:
        """Alias for :attr:`transform` (the branch parameter transform).

        ``run.py``/moves refer to the branch's parameter transform as
        ``transform_fn`` while the :class:`Settings` field is named
        ``transform``; expose both names for the same object so either spelling
        resolves on any :class:`Setup`.
        """
        return self.transform

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
    priors: Optional[typing.Dict[str, ProbDistContainer]] = None
    periodic: Optional[dict] = None
    nleaves_max: Optional[int] = None
    nleaves_min: Optional[int] = None
    ndim: Optional[int] = None
    # This branch's OWN temperature count (each module tempers internally;
    # the engine runs cold-chain only). An explicit ``betas`` ladder wins
    # and defines ntemps = len(betas); otherwise ``ntemps`` sizes the
    # default ladder the branch's move builder constructs.
    ntemps: Optional[int] = None
    betas: Optional[np.ndarray] = None
    other_tempering_kwargs: Optional[dict] = None
    branch_state: Optional[eryn_State] = None
    branch_backend: Optional[eryn_Backend] = None
    log_dir: Optional[str] = None
    # Per-branch params-based template generator registered into every
    # AnalysisContainer's dictionary ``signal_gen`` by ``setup_acs``
    # (run.py). Called per leaf as ``fn(*sampling_params) -> DomainBase``
    # and must therefore wrap the sampling->waveform transform AND the
    # waveform generator + domain projection. With one registered, the
    # engine builds/subtracts this branch's templates from the residuals
    # under the hood (``rebuild_residuals``) -- no ``_build_*`` template
    # loops or ``subtract_initial_signal`` calls needed in the settings
    # file's recipe.
    signal_gen: Optional[Callable] = None


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
    num_iterations: int | None = 500
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
    random_seed: int | None = None
    backup_iter: int | None = None
    # HDF backend chain compression (parallel-resources plan P2): gzip-4 by
    # default — level 9 costs the saver a lot of CPU for marginal size gains.
    hdf_compression: str = "gzip"
    hdf_compression_opts: int = 4
    nwalkers: int | None = None
    ntemps: int | None = None
    window_type: str = "tukey"
    window_taper_duration: float | None = None
    # Run-level likelihood convention: when True, every AnalysisContainer the
    # engine builds defaults its likelihood methods to source_only=True
    # (log L = -1/2 <r|r>, no -sum(log|detC|) noise normalization term). ONLY
    # for runs with a FIXED PSD (no psd sampling branch): there the noise
    # term is an overall constant, so likelihood differences / acceptances
    # are unchanged while the readout becomes directly interpretable as the
    # residual inner product. Runs that sample the PSD need the noise term
    # and must keep this False.
    likelihood_source_only: bool = False
    gpu_backend: str = "cuda12x"
    gpus: typing.List[int] | None = None
    fixed_psd_kwargs: typing.Dict[str, typing.Any] | None = None
    data_processor_class: Optional[BaseProcessingStep] = None
    processor_init_kwargs: Optional[dict] = None
    preprocess_kwargs: Optional[dict] = None
    sensitivity_init_kwargs: Optional[dict] = None
    # Class used to build ``self.sensitivity_backend``. Defaults to
    # :class:`CompositeSensitivityBackend`, which yields a
    # :class:`CompositeSensitivityMatrix` per walker (InstrumentNoise plus
    # optional GalacticForeground / SGWB components). Set to
    # :class:`XYZSensitivityBackend` for the legacy C++/CUDA matrix path.
    sensitivity_backend_class: Optional[type] = None
    normalize_window: bool = False
    catalogue: typing.Optional[dict] = None
    # Run-level console verbosity: False (default) keeps the console quiet —
    # log statements go to the log files only, progress bars are off, and the
    # stock moves/processors stay silent. True streams the logs to stdout and
    # turns the progress bars back on. Stock fits expose this as a headline
    # knob (erebor.<variant>(verbose=True) / env VERBOSE).
    verbose: bool = False
    # Progress bar ONLY, decoupled from log streaming. ``None`` (default)
    # follows ``verbose``, preserving the historical pairing; ``True`` turns
    # the sampler's tqdm bar on while the console stays quiet, ``False``
    # suppresses the bar even under ``verbose=True`` (wanted when stdout is a
    # log file). Stock fits expose it as ``progress=`` / env ``PROGRESS``.
    progress: Optional[bool] = None

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
    """Setup object that ingests data, prepares orbits, and builds the sensitivity backend.

    On construction it:

    1. Ensures the artifacts directory exists and attaches a logger.
    2. Runs the configured :attr:`data_processor` to load and condition the
       time-domain data.
    3. Resolves ``domain_settings`` (instance or factory) into a concrete
       :class:`~lisatools.domains.DomainSettingsBase`, builds the analysis
       window, and constructs the input :class:`DataResidualArray`.
    4. Configures :class:`XYZSensitivityBackend` for use in PSD/likelihood
       calls.

    Args:
        general_settings: The :class:`GeneralSettings` to use.
    """

    def __init__(self, general_settings: GeneralSettings):

        Setup.__init__(self, general_settings)

        level = logging.DEBUG
        name = "GeneralSetup"
        # exist_ok: several MPI ranks build concurrently (run_global.py builds
        # on every rank) — check-then-create here is a startup race.
        os.makedirs(self.artifacts_file_dir, exist_ok=True)
        self.logger = init_logger(
            filename="general_setup.log", level=level, name=name,
            log_dir=self.artifacts_file_dir,
            console=bool(getattr(self, "verbose", False)),
        )

        self.init_setup()

    @property
    def main_file_path(self) -> str:
        """Filesystem path of the primary HDF5 backend file."""
        # os.path.join, NOT string concat: FILE_STORE_DIR without a trailing
        # slash must not silently rename the run (observed on a cluster
        # relaunch: '<dir>global_fit_..._artifacts' beside the real run dir).
        return os.path.join(
            self.file_store_dir, self.base_file_name + "_" + self.main_file_key + ".h5"
        )

    @property
    def artifacts_file_dir(self) -> str:
        """Directory where logs and diagnostic plots are stored for this run."""
        return os.path.join(self.file_store_dir, self.base_file_name + "_artifacts/")

    @property
    def data_t0(self) -> float:
        return self.data_td_settings.t0

    @property
    def data_dt(self) -> float:
        return self.data_td_settings.dt

    def _resolve_deferred_noise_model(self, sensitivity_init_kwargs: dict) -> dict:
        """Resolve deferred noise-model specs and record the model identity.

        Variant ``finalize_general`` code runs before the data are processed,
        so it cannot know the data epoch. Two deferred spellings — carrying
        only plain scalars/paths on the settings tree — are resolved here,
        where ``data_t0`` is authoritative:

        * ``instrument_component_kwargs["ltts_l1_file"]`` (+ optional
          ``"ltts_stride"``): replaced by a
          :class:`~lisatools.sensitivity.LinkDelayTable` read from that L1
          brick's ``/ltts`` group, anchored at ``data_t0``.
        * ``galfor_modulation_anchor="data_t0"``: rewrites the lazy galfor
          modulation table's epoch to ``data_t0``.

        Every :class:`~lisatools.sensitivity.GalForTimeModulation` is also
        coverage-checked against the observation span, failing loudly at
        build instead of interpolating nonsense at first proposal.

        Side effect: sets ``self.noise_model_identity`` — the semantic noise
        model identity (instrument class, WDM PSD method, delay-table and
        modulation digests, data epoch) the run backend persists so a resume
        under a silently different likelihood is refused.
        """
        from ..sensitivity import GalForTimeModulation, LinkDelayTable

        def _digest(*arrays) -> str:
            h = hashlib.sha256()
            for a in arrays:
                a = np.ascontiguousarray(np.asarray(a, dtype=float))
                h.update(str(a.shape).encode())
                h.update(a.tobytes())
            return h.hexdigest()[:16]

        comp_kwargs = sensitivity_init_kwargs.get("instrument_component_kwargs")
        ltts_digest = None
        if isinstance(comp_kwargs, dict) and "ltts_l1_file" in comp_kwargs:
            comp_kwargs = dict(comp_kwargs)
            l1_path = comp_kwargs.pop("ltts_l1_file")
            stride = int(comp_kwargs.pop("ltts_stride", 200))
            table = LinkDelayTable.from_l1_file(
                l1_path, stride=stride, data_t0=self.data_t0
            )
            step = float(table.t[1] - table.t[0]) if table.t.size > 1 else 0.0
            if (
                float(table.t[0]) - step > self.data_t0
                or float(table.t[-1]) + step < self.data_t0 + self.Tobs
            ):
                raise ValueError(
                    f"[unequal-arm] delay table {l1_path} spans "
                    f"[{float(table.t[0]):.6e}, {float(table.t[-1]):.6e}] s "
                    f"(mission clock) but the data occupy "
                    f"[{self.data_t0:.6e}, {self.data_t0 + self.Tobs:.6e}] s "
                    "-- wrong file or wrong epoch."
                )
            comp_kwargs["ltts"] = table
            sensitivity_init_kwargs["instrument_component_kwargs"] = comp_kwargs
            ltts_digest = _digest(table.t, table.ltts)
            self.logger.info(
                "[unequal-arm] link-delay table %s: stride=%d, %d epochs over "
                "[%.6e, %.6e] s (mission clock), anchored at data_t0=%.6e, "
                "digest=%s",
                l1_path,
                stride,
                table.t.size,
                float(table.t[0]),
                float(table.t[-1]),
                self.data_t0,
                ltts_digest,
            )

        mod_anchor = sensitivity_init_kwargs.pop("galfor_modulation_anchor", None)
        mod = sensitivity_init_kwargs.get("galfor_modulation")
        mod_path = None
        mod_digest = None
        if mod_anchor is not None:
            if mod_anchor != "data_t0":
                raise ValueError(
                    "galfor_modulation_anchor must be 'data_t0' or absent; "
                    f"got {mod_anchor!r}."
                )
            if not isinstance(mod, GalForTimeModulation):
                raise ValueError(
                    "galfor_modulation_anchor='data_t0' requires a tabulated "
                    "galfor modulation (set GALFOR_MODULATION_PATH / "
                    "general.galfor_modulation_path)."
                )
            mod.t0 = float(self.data_t0)
            self.logger.info(
                "[galfor-modulation] table epoch anchored at data_t0=%.6e",
                self.data_t0,
            )
        if isinstance(mod, GalForTimeModulation):
            tbl = mod._table()
            mod_path = os.path.basename(mod.path)
            mod_digest = _digest(tbl)
            t_rel = tbl[:, 0] - mod.t0
            step = float(t_rel[1] - t_rel[0]) if t_rel.size > 1 else 0.0
            if t_rel[0] - step > 0.0 or t_rel[-1] + step < self.Tobs:
                raise ValueError(
                    f"galfor modulation table {mod.path} covers "
                    f"[{t_rel[0]:.6e}, {t_rel[-1]:.6e}] s relative to its "
                    f"epoch (t0={mod.t0:.6e}) but the data span "
                    f"[0, {self.Tobs:.6e}] s. A table on the absolute mission "
                    "clock needs galfor_modulation_t0='data' "
                    "(GALFOR_MODULATION_T0=data) or an explicit epoch."
                )
            self.logger.info(
                "[galfor-modulation] %s: %d epochs covering "
                "[%.6e, %.6e] s of the data frame (t0=%.6e), digest=%s",
                mod.path,
                tbl.shape[0],
                float(t_rel[0]),
                float(t_rel[-1]),
                mod.t0,
                mod_digest,
            )

        comp_kwargs_now = sensitivity_init_kwargs.get("instrument_component_kwargs") or {}
        if ltts_digest is None and "ltts" in comp_kwargs_now:
            lt = comp_kwargs_now["ltts"]
            ltts_digest = (
                _digest(lt.t, lt.ltts)
                if isinstance(lt, LinkDelayTable)
                else _digest(lt)
            )
        inst_cls = sensitivity_init_kwargs.get("instrument_component_cls")
        wdm_method = (
            sensitivity_init_kwargs.get("wdm_psd_method")
            or comp_kwargs_now.get("wdm_psd_method")
            or "fold"
        )
        self.noise_model_identity = {
            "instrument_component": (
                inst_cls.__name__ if isinstance(inst_cls, type) else str(inst_cls)
            )
            if inst_cls is not None
            else "InstrumentNoise",
            "unequal_arm": bool(ltts_digest is not None),
            "wdm_psd_method": str(wdm_method),
            "ltts_digest": ltts_digest or "",
            "galfor_modulation": mod_path or "",
            "galfor_modulation_digest": mod_digest or "",
            "galfor_modulation_t0": float(getattr(mod, "t0", 0.0) or 0.0),
            "data_t0": float(self.data_t0),
        }
        self.logger.info("[noise-model-identity] %s", self.noise_model_identity)
        return sensitivity_init_kwargs

    # NOTE: ``catalogue`` is a plain attribute (a ``GeneralSettings``
    # dataclass field re-applied by ``Setup.__init__`` and refreshed from
    # ``data_processor.catalogue`` in ``init_data_information``) — it must
    # NOT be a read-only property here or the dataclass-field setattr fails.

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
        os.makedirs(self.artifacts_file_dir, exist_ok=True)

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

    def init_data_information(self):
        """Run preprocessing, build the basis domain, and configure the sensitivity backend."""
        if self.data_processor_class is None:
            raise ValueError("Must provide data_processor_class for GeneralSetup.")

        self.data_processor = self.data_processor_class(**(self.processor_init_kwargs or {}))

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

        times, _ = self.data_processor.process(**self.preprocess_kwargs)
        dt = self.data_processor.td_signal.settings.dt
        Nt = len(times)
        self.catalogue = getattr(self.data_processor, 'catalogue', {})
        # TD settings of the loaded data (stft_tof side): consumed by the
        # TDWaveformBase-family wrappers as ``data_td_settings``. NOTE:
        # ``TDSettings.args`` is only ``(N, dt)`` — ``t0`` must be carried
        # explicitly or the loader's start time (the data_t0 anchor the
        # downstream waveform wrappers align their grids to) is lost.
        _loader_settings = self.data_processor.td_signal.settings
        self.data_td_settings = TDSettings(
            *_loader_settings.args,
            t0=float(times[0]),
            force_backend=self.force_backend,
        )
        # ``data_t0`` is the read-only property data_td_settings.t0.
        assert abs(self.data_t0 - float(times[0])) < 1e-9
        self.Tobs = Nt * dt

        domain_settings = self._resolve_domain_settings(times=times, dt=dt)
        self.basis_kwargs = dict(force_backend=self.force_backend)

        # Domain-specific window length + diagnostic-plot kwargs. The window
        # is always built on the underlying time grid (length Nt); the basis
        # transform inside data_processor.pour() handles the projection.
        # Dispatch is by DomainSettingsBase child class -- never a string flag.
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
        self.input_data_residual_array, orbits = self.data_processor.pour(
            settings=domain_settings, window=window, return_orbits=True
        )

        if isinstance(domain_settings, FDSettings):
            # FD path: ``pour`` returns an FDSignal whose ``settings`` IS the
            # resolved ``domain_settings`` instance, so the active-band
            # ``f_arr`` / ``df`` / ``start_freq_ind`` are already carried by
            # the DomainBase (the legacy DataResidualArray
            # ``_store_time_and_frequency_information`` call is gone).
            # ``data_length`` is kept as a plain attribute for moves that
            # still key off it.
            self.input_data_residual_array.data_length = len(domain_settings.f_arr)

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
            orbits_kwargs = orbits.kwargs

            if self.force_backend == self.gpu_backend:
                # Rebuild on the GPU backend via the orbits' reproducibility
                # properties (stft_tof); ``kwargs`` carries armlength, t0 and
                # frame, so ICRS orbit files round-trip correctly.
                orbits_kwargs["force_backend"] = self.gpu_backend
                self.logger.debug(f"Initializing GPU orbits with kwargs: {orbits_kwargs}")

                self.gpu_orbits = self.data_processor.orbits_class(*orbits.args, **orbits_kwargs)

        self.init_orbit_information()

        # Sensitivity backend: defaults to CompositeSensitivityBackend, which
        # builds a CompositeSensitivityMatrix (InstrumentNoise + optional
        # GalacticForeground / SGWB components) per walker. Set
        # ``sensitivity_backend_class=XYZSensitivityBackend`` for the C++/CUDA
        # matrix path (incl. the stft_tof galactic-grid foreground).
        sensitivity_init_kwargs = dict(self.sensitivity_init_kwargs or {})

        # stft_tof: anchor the galactic grid's orbit-phase reference to the
        # data start when the caller did not set it explicitly.
        if "galactic_grid_kwargs" in sensitivity_init_kwargs and isinstance(
            sensitivity_init_kwargs["galactic_grid_kwargs"], dict
        ):
            sensitivity_init_kwargs["galactic_grid_kwargs"].setdefault("t0", self.data_t0)

        # Late resolution of mission-clock noise-model inputs. finalize-time
        # variant code cannot know the data epoch (the authoritative
        # ``data_t0`` exists only after ``init_data_information``), so the
        # stock variants pass deferred specs resolved here, exactly like the
        # galactic-grid ``t0`` default above. Only plain scalars/paths ride
        # the settings tree; the heavy/tabulated objects are built now.
        sensitivity_init_kwargs = self._resolve_deferred_noise_model(
            sensitivity_init_kwargs
        )

        backend_cls = self.sensitivity_backend_class or CompositeSensitivityBackend
        if backend_cls is CompositeSensitivityBackend or (
            isinstance(backend_cls, type)
            and issubclass(backend_cls, CompositeSensitivityBackend)
        ):
            # Drop kwargs that are specific to the XYZ backend so a
            # settings file can be switched over without editing every kwarg.
            xyz_only = ("mask_percentage", "use_splines", "spline_order", "galactic_grid_kwargs")
            for k in xyz_only:
                if k in sensitivity_init_kwargs:
                    self.logger.debug(
                        f"Ignoring XYZ-only sensitivity kwarg {k!r} for "
                        "CompositeSensitivityBackend."
                    )
                    sensitivity_init_kwargs.pop(k)
            self.sensitivity_backend = backend_cls(
                settings=domain_settings,
                force_backend=self.force_backend,
                **sensitivity_init_kwargs,
            )
        else:
            self.sensitivity_backend = backend_cls(
                orbits=self.gpu_orbits,
                settings=domain_settings,
                force_backend=self.force_backend,
                window_values=window if self.normalize_window else None,
                **sensitivity_init_kwargs,
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
    setup_function: typing.Callable[(...), None]
    source_metadata: typing.Dict[str, dataclasses.dataclass] = dataclasses.field(
        default_factory=dict
    )


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
