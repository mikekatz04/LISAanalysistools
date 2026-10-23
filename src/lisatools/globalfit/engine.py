"""Engine and configuration dataclasses for the LISA global-fit sampler.

What lives here:

* the per-branch knob layer and its built twin, :class:`Settings` /
  :class:`Setup` (every stock branch block subclasses them);
* the run-wide pair :class:`GeneralSettings` / :class:`GeneralSetup`;
  building a :class:`GeneralSetup` loads and conditions the data (the
  configured ``data_processor_class``), resolves the analysis domain, builds
  the analysis window, the data signal in that domain (a
  :class:`~lisatools.domains.DomainBase` child, ``WDMSignal`` on the
  production grid) and the sensitivity backend, and resolves the deferred
  noise-model inputs
  (unequal-arm link-delay table, galfor modulation epoch);
* the bundles :class:`GlobalFitSettings` (what
  :class:`~lisatools.globalfit.run.GlobalFitSetup` is built from),
  :class:`EngineInfo` and :class:`RankInfo`;
* the sampler :class:`GlobalFitEngine`, an eryn
  :class:`~eryn.ensemble.EnsembleSampler` that carries the shared
  :class:`~lisatools.analysiscontainer.AnalysisContainerArray`, and its model
  tuple :data:`GlobalFitInfo`.

Pipeline position: ``StockGlobalFit.build`` (:mod:`lisatools.globalfit.stock`)
builds the :class:`GeneralSetup`, one :class:`Setup` per branch and then the
:class:`~lisatools.globalfit.run.GlobalFitSetup`.
:meth:`~lisatools.globalfit.run.GlobalFit.prepare_main` builds the head's
:class:`GlobalFitEngine` (store-backed, the recipe as its ``stopping_fn``);
:meth:`~lisatools.globalfit.run.GlobalFit.prepare_compute` builds an
in-memory shell engine on each compute rank. In the stock fits the engine
runs the cold chain only (``general.ntemps`` is pinned to 1) and every branch
tempers on its own ladder inside its sub-state; ``erebor.blank`` is the
exception (engine ladder ``NTEMPS``, default 2, for its simple-API branches).

Note:
    ``__all__`` names only :data:`GlobalFitInfo` and :class:`GlobalFitEngine`,
    so ``automodule :members:`` in the Sphinx docs renders only those two; the
    other classes are imported directly (``from lisatools.globalfit.engine
    import GeneralSettings, Settings, Setup``).
"""

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
            (the highest non-head rank; see
            :func:`lisatools.globalfit.communication.ranks.resolve_roles`).
        main_rank: Rank (of the run communicator) that becomes the HEAD: it
            drives the sampler loop and computes walker block 0. The stock
            fits pass ``0``; the dataclass default ``-1`` is a placeholder.
    """

    head_rank: int = -1
    main_rank: int = -1


class Setup:
    """Adapter wrapping a :class:`Settings` dataclass for use as a setup object.

    A ``*Setup`` is the BUILT twin of its ``*Settings``: construction copies
    every dataclass field of the settings onto ``self`` (by re-running the
    settings class's dataclass ``__init__`` on this object), keeps the
    original on :attr:`settings`, and rounds ``Tobs`` with :meth:`init_df`
    (so ``Tobs`` and ``dt`` must be set). The stock branch setups
    (``GBSetup``, ``MBHSetup``, ``PSDSetup``, ...) subclass it.

    Args:
        settings_holder: A :class:`Settings` instance whose fields populate
            this :class:`Setup`. The dataclass field metadata is introspected
            so the same constructor can be re-applied.

    Attributes:
        Tobs (float): Observation time rounded down to a multiple of ``dt``
            (s).
        df (float): ``1 / Tobs`` (Hz).
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
        """Round ``Tobs`` down to a multiple of ``dt`` and recompute ``df = 1/Tobs``."""
        self.Tobs = int(self.Tobs / self.dt) * self.dt
        self.df = 1.0 / self.Tobs


@dataclasses.dataclass
class Settings:
    """Common configuration fields shared by branch-specific setup objects.

    Most fields default to ``None`` so subclasses can extend or override
    them; users assemble settings either by subclassing or by setting fields
    directly. The stock branch blocks (``fit.gb``, ``fit.mbh``, ...) are
    subclasses whose defaults resolve from env knobs named after the field
    (``GB_NTEMPS`` for ``gb.ntemps``).

    Args:
        Tobs: Observation time (s).
        dt: Sampling cadence (s).
        initialize_kwargs: Branch-specific construction kwargs.
        transform: Sampling-basis -> waveform-basis
            :class:`~eryn.utils.transform.TransformContainer`.
        priors: ``{branch: ProbDistContainer}`` priors in the sampling basis.
        periodic: ``{branch: {parameter name or index: period}}``; names are
            resolved through ``transform.input_basis``.
        nleaves_max: Maximum leaves (sources) on the branch.
        nleaves_min: Minimum leaves; equal to ``nleaves_max`` for a fixed-leaf
            branch (always on at a fresh start).
        ndim: Parameters per leaf (sampling basis).
        ntemps: This branch's OWN temperature count (in the stock fits the
            engine runs the cold chain only). An explicit ``betas`` wins and
            defines ``ntemps = len(betas)``; with neither, the branch runs on
            the engine ladder.
        betas: Explicit inverse-temperature ladder ``(ntemps,)``.
        other_tempering_kwargs: Extra kwargs for the branch's tempering.
        branch_state: The sub-state class of the branch (e.g.
            :class:`~lisatools.globalfit.state.GBState`), instantiated per
            branch by :class:`~lisatools.globalfit.state.GFState`.
        branch_backend: The matching sub-backend class (e.g.
            :class:`~lisatools.globalfit.hdfbackend.GBHDFBackend`).
        log_dir: Directory for the branch's log file.
        signal_gen: Params-based template generator ``fn(*sampling_params) ->
            DomainBase`` (wraps transform + waveform + domain projection),
            registered in every container's ``signal_gen`` dict by
            :meth:`~lisatools.globalfit.run.GlobalFit.setup_acs`, which then
            subtracts the branch's templates from the residuals itself.
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
    :meth:`GeneralSetup.init_setup` for the checks). The stock fits expose
    this block as ``fit.general`` (a subclass whose defaults resolve from env
    knobs named after the field: ``NUM_ITERATIONS``, ``NWALKERS``,
    ``BACKUP_ITER``, ``MIDIT_CHECKPOINT``, ``VERBOSE``, ``PROGRESS``, ...) and
    add run fields of their own (``data_mode``, ``coarse_Q``,
    ``make_diagnostic_plots``, ``plot_iterations``, ...).

    Args:
        num_iterations: Sampler iterations for ``run_mcmc`` /
            :meth:`~lisatools.globalfit.run.GlobalFit.sample`. Default 500.
        Tobs: Requested observation time (s); after the data load it is reset
            to the loaded span ``Nt * dt``.
        dt: Sampling cadence (s).
        file_store_dir: Output directory (required).
        base_file_name: Run name stem (required); the store is
            ``<file_store_dir>/<base_file_name>_<main_file_key>.h5`` and the
            artifacts go to ``<base_file_name>_artifacts/``.
        main_file_key: Store-name suffix. Default
            ``"parameter_estimation_main"``.
        past_file_for_start: Optional store to restart from directly (its
            last sample) when the run's own store has no saved iteration.
        orbits: Detector orbits. Orbits returned by the data processor's
            ``pour`` (the L1 data's own orbits on the mojito path) replace
            this field (and are rebuilt on the GPU backend as ``gpu_orbits``
            on a GPU run); ``None`` with no processor orbits falls back to
            :class:`~lisatools.detector.EqualArmlengthOrbits`.
        gpu_orbits: The same orbits on the GPU backend (required alongside a
            user-supplied ``orbits`` on a GPU run).
        domain_settings: A :class:`~lisatools.domains.DomainSettingsBase`
            instance, or a factory ``(times, dt, force_backend) ->
            DomainSettingsBase`` called after the data load (the STFT path).
            Required; dispatch is by class, never by string.
        random_seed: Run seed. Multi-rank: the seed base every rank derives
            its RNG stream from (``None`` -> fresh entropy drawn on the head).
        backup_iter: Saver-rank cadence, in saves, of the running backup copy
            of the store (``None``, the dataclass default, means every save;
            the stock default is 1; the 9mo/12mo launchers export
            ``BACKUP_ITER=10``).
        midit_checkpoint: Arm the mid-iteration checkpoint
            (:mod:`lisatools.globalfit.midit_checkpoint`) on the head. Default
            ``True``; the 9mo/12mo launchers export ``MIDIT_CHECKPOINT=0``.
        midit_checkpoint_min_interval: Minimum seconds between checkpoint
            writes. Default 600.
        hdf_compression: Store dataset compression. Default ``"gzip"``.
        hdf_compression_opts: Compression level. Default 4.
        nwalkers: Walkers (the stock fits: env ``NWALKERS``; production 4).
        ntemps: Engine temperature count (the stock fits pin it to 1;
            ``erebor.blank`` defaults to 2).
        window_type: Analysis window passed to
            :func:`~lisatools.utils.utility.windowfun`. Default ``"tukey"``.
        window_taper_duration: Taper duration (s); the window's ``alpha`` is
            ``window_taper_duration / (N * dt)`` with ``N`` the data length
            (the segment length on an STFT grid). Must be set.
        likelihood_source_only: Every container defaults to the source-only
            likelihood ``-1/2 <r|r>`` (no noise normalization). Only valid for
            a FIXED PSD; ignored (with a warning) when a ``psd`` branch exists.
        gpu_backend: Backend name used when ``gpus`` is set. Default
            ``"cuda12x"``.
        gpus: Device ids of this rank (after per-rank pinning); ``None`` runs
            on the CPU backend.
        fixed_psd_kwargs: Sensitivity kwargs when no ``psd`` branch is
            sampled; ``galfor_params`` here also pins the foreground when a
            ``psd`` branch exists without a ``galfor`` branch. ``None`` ->
            ``{"psd_params": [15e-12, 3e-15], "galfor_params": None}``.
        data_processor_class: Data loader/conditioner class (a
            :class:`~lisatools.globalfit.preprocessing.BaseProcessingStep`
            subclass; required).
        processor_init_kwargs: Its constructor kwargs.
        preprocess_kwargs: Merged over the defaults (high-pass
            ``cutoff=2e-5, order=2, zero_phase=True``; trim 200 h from each
            end; ``Tobs``; plot folder) and passed to its ``process``.
        sensitivity_init_kwargs: Sensitivity-backend constructor kwargs
            (deferred noise-model specs resolved at build, see
            :meth:`GeneralSetup._resolve_deferred_noise_model`).
        sensitivity_backend_class: Backend class; ``None`` ->
            :class:`~lisatools.sensitivity.CompositeSensitivityBackend`.
        normalize_window: Pass the window to a non-composite (XYZ) backend as
            ``window_values``.
        catalogue: Source catalogue; refreshed from the data processor after
            the load.
        verbose: Stream logs to the console and turn progress bars on.
            Default ``False`` (logs go to files only).
        progress: Sampler progress bar alone; ``None`` follows ``verbose``.
        global_fit_codename: Run metadata, propagated to
            :class:`~lisatools.globalfit.postprocessing.RunMetadata` (the same
            holds for the remaining metadata fields below).
        global_fit_version: Run metadata.
        global_fit_contact: Run metadata.
        global_fit_code_link: Run metadata.
        submission_parent_folder: Output folder of the end-of-run submission
            (:class:`~lisatools.globalfit.postprocessing.SubmissionWriter`);
            ``None`` writes no submission.
        input_data_link: Run metadata.
        input_reference: Run metadata.
        noise_model: Run metadata.
        noise_model_code_link: Run metadata.
        run_waveform_model: Run metadata.
        run_waveform_model_code_link: Run metadata.
        comment: Run metadata.
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
    # Mid-iteration checkpointing (preemption protection): persist the
    # in-memory state at sub-iteration boundaries (between a combine
    # stage's moves; between PE leaves) so a killed job resumes from the
    # newest snapshot instead of losing the whole iteration. See
    # lisatools.globalfit.midit_checkpoint for the resume semantics.
    midit_checkpoint: bool = True
    # Minimum seconds between checkpoint writes (the worst-case overhead
    # is one state pickle per interval; raise it for runs with very large
    # states, e.g. full-galaxy GB fits).
    midit_checkpoint_min_interval: float = 600.0
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

    On construction (``StockGlobalFit.build_general`` on every rank) it:

    1. Ensures the artifacts directory exists and attaches a logger
       (``general_setup.log``).
    2. Checks the required fields and picks ``force_backend``
       (``gpu_backend`` when ``gpus`` is set, else ``"cpu"``).
    3. Builds ``data_processor_class(**processor_init_kwargs)`` and runs its
       ``process(**preprocess_kwargs)`` to load and condition the time-domain
       data; ``Tobs`` becomes the loaded span.
    4. Resolves ``domain_settings`` (instance or factory) into a concrete
       :class:`~lisatools.domains.DomainSettingsBase`, builds the analysis
       window, and pours the data into that domain (the processor's ``pour``
       returns a :class:`~lisatools.domains.DomainBase` child such as
       ``WDMSignal``), plus diagnostic data plots.
    5. Takes the processor's orbits when it returns them, resolves the
       deferred noise-model inputs (:meth:`_resolve_deferred_noise_model`)
       and builds the sensitivity backend: ``sensitivity_backend_class``,
       default :class:`~lisatools.sensitivity.CompositeSensitivityBackend`
       (XYZ-only kwargs are dropped for it).

    Args:
        general_settings: The :class:`GeneralSettings` to use.

    Attributes:
        force_backend (str): ``"cpu"`` or the GPU backend name.
        data_processor: The built data processor.
        data_td_settings (TDSettings): Time-domain settings of the loaded data,
            carrying its start time ``t0`` (see :attr:`data_t0`).
        domain_settings (DomainSettingsBase): The resolved analysis domain.
        input_data_residual_array (DomainBase): The data in the analysis
            domain (``WDMSignal`` on a WDM grid); every walker's container
            starts from a deep copy of it.
        sensitivity_backend: Callable building one walker's sensitivity
            matrix from noise parameters.
        noise_model_identity (dict): Semantic noise-model identity persisted
            by the run backend (resume refuses a mismatch).
        catalogue (dict): The processor's source catalogue.
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
        """Start time of the loaded data (s, mission clock): the epoch every grid aligns to."""
        return self.data_td_settings.t0

    @property
    def data_dt(self) -> float:
        """Sampling cadence of the loaded data (s)."""
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
        under a silently different likelihood is refused — and publishes the
        resolved kwargs as ``self.sensitivity_init_kwargs``.

        Args:
            sensitivity_init_kwargs (dict): A copy of the configured
                sensitivity-backend kwargs, possibly carrying the deferred
                specs above.

        Returns:
            dict: The resolved kwargs (``ltts`` table in place of
            ``ltts_l1_file`` / ``ltts_stride``; ``galfor_modulation_anchor``
            removed).

        Raises:
            ValueError: The delay table or the modulation table does not cover
                the data span, or ``galfor_modulation_anchor`` is not
                ``"data_t0"`` / has no tabulated modulation to anchor.
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
            # Coarse-likelihood identity (plan-2): a resume across a coarse
            # mode/Q/weighting change is a different PSD/GALFOR kernel on
            # identical shapes. The fiducial digest is appended at write time
            # by run.py once the sidecar runtime exists.
            "coarse_mode": str(getattr(self, "coarse_gpu_mode", "off") or "off"),
            "coarse_Q": int(getattr(self, "coarse_Q", 1) or 1),
            "coarse_use_ws": bool(getattr(self, "coarse_use_ws", True)),
        }
        self.logger.info("[noise-model-identity] %s", self.noise_model_identity)
        # Publish the RESOLVED kwargs as the single source of truth. This
        # method works on a copy, so without the write-back the attribute
        # keeps the DEFERRED spec -- ``galfor_modulation_anchor`` plus
        # ``instrument_component_kwargs{ltts_l1_file, ltts_stride}`` -- and any
        # SECOND consumer that rebuilds a backend from it gets keys no
        # constructor accepts. The coarse sidecar in
        # ``run.py::_prepare_coarse_wdm_runtime`` is exactly such a consumer
        # (job 372: "CompositeSensitivityBackend.__init__() got an unexpected
        # keyword argument 'galfor_modulation_anchor'"), and it would have hit
        # the unresolved ltts spec immediately afterwards.
        self.sensitivity_init_kwargs = sensitivity_init_kwargs
        return sensitivity_init_kwargs

    # NOTE: ``catalogue`` is a plain attribute (a ``GeneralSettings``
    # dataclass field re-applied by ``Setup.__init__`` and refreshed from
    # ``data_processor.catalogue`` in ``init_data_information``) — it must
    # NOT be a read-only property here or the dataclass-field setattr fails.

    def init_setup(self):
        """Validate required settings, pick ``force_backend`` and run :meth:`init_data_information`.

        Raises:
            ValueError: ``file_store_dir``, ``base_file_name`` or
                ``domain_settings`` is missing.
        """
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
        """Fill in missing orbit objects (fallback :class:`EqualArmlengthOrbits`).

        Runs after the data processor's orbits (if any) were installed. With
        ``orbits`` still ``None`` both fields get equal-arm orbits (the GPU
        copy on ``force_backend``).

        Raises:
            ValueError: ``orbits`` is set but ``gpu_orbits`` is ``None`` on a
                GPU run.
        """
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

        Args:
            times: Time stamps of the loaded data (s).
            dt: Sampling cadence (s).

        Returns:
            DomainSettingsBase: The resolved domain settings.

        Raises:
            TypeError: ``domain_settings`` is neither.
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
        """Run preprocessing, build the basis domain, and configure the sensitivity backend.

        Steps 3-5 of the class description. Sets ``data_processor``,
        ``fixed_psd_kwargs`` (default when ``None``), ``preprocess_kwargs``
        (merged over the defaults), ``catalogue``, ``data_td_settings``,
        ``Tobs``, ``domain_settings``, ``window_alpha``,
        ``input_data_residual_array``, the orbits, ``noise_model_identity``
        and ``sensitivity_backend``.

        Raises:
            ValueError: No ``data_processor_class``.
            NotImplementedError: The resolved domain is not STFT, FD or WDM.
        """
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

    What :class:`~lisatools.globalfit.run.GlobalFitSetup` is built from
    (``StockGlobalFit.build`` fills it with the built setups).

    Args:
        source_info: Mapping of branch name to its built :class:`Setup`.
        general_info: The built :class:`GeneralSetup`.
        rank_info: MPI rank assignments.
        setup_function: Callback that materializes the recipe's moves, called
            once per rank by
            :meth:`~lisatools.globalfit.run.GlobalFit._build_acs_and_recipe` as
            ``setup_function(recipe, engine_info, curr, acs, priors, state)``
            (the stock variants pass their module-level ``setup_recipe``).
        source_metadata: Extra run objects keyed by name; the stock fits put
            the live :class:`~lisatools.globalfit.recipe.Recipe` under
            ``"recipe"`` (``None`` there -> an empty recipe).
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
    """Branch-keyed metadata that the engine needs in order to launch a sampler.

    Built by :attr:`lisatools.globalfit.run.GlobalFitSetup.engine_info` from
    the branch setups.

    Args:
        branch_names: Branch names in model order.
        ndims: ``{branch: parameters per leaf}``.
        nleaves_max: ``{branch: maximum leaves}``.
        nleaves_min: ``{branch: minimum leaves}``.
        branch_states: ``{branch: sub-state class or None}`` (the
            ``sub_state_bases`` of :class:`~lisatools.globalfit.state.GFState`).
        branch_backends: ``{branch: sub-backend class or None}``.
    """

    branch_names: typing.List[str]
    ndims: typing.Dict[str, int]
    nleaves_max: typing.Dict[str, int]
    nleaves_min: typing.Dict[str, int]
    branch_states: typing.Dict[str, eryn_State] = None
    branch_backends: typing.Dict[str, eryn_Backend] = None


#: The ``model`` every global-fit move receives (from
#: :meth:`GlobalFitEngine.get_model` on the head; built by hand on a compute
#: rank): ``analysis_container_arr`` (this rank's
#: :class:`~lisatools.analysiscontainer.AnalysisContainerArray`, the residuals
#: the moves score against), ``map_fn`` (the pool's ``map`` or the builtin) and
#: ``random`` (the RNG, a ``numpy.random.RandomState``). Unlike eryn's ``Model``
#: it carries no likelihood or prior callables.
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
    during the run. It samples ``general.ntemps`` rungs (``tempering_kwargs``),
    i.e. the COLD chain only in the stock fits (``erebor.blank`` aside); the moves come
    from the recipe (each step's ``setup_run`` sets ``sampler.moves``) and the
    :class:`~lisatools.globalfit.recipe.Recipe` is the ``stopping_fn``, called
    after every iteration. Built by
    :meth:`~lisatools.globalfit.run.GlobalFit.prepare_main` (head, with the
    run's :class:`~lisatools.globalfit.hdfbackend.GFHDFBackend`) and
    :meth:`~lisatools.globalfit.run.GlobalFit.prepare_compute` (compute-rank
    shell: in-memory backend, never sampled).

    Args:
        analysis_container_arr: Shared :class:`AnalysisContainerArray` for
            this run (this rank's walker block under the multi-rank layout).
        *args: Forwarded to :class:`eryn.ensemble.EnsembleSampler`
            (``nwalkers, ndims, log_like_fn, priors``).
        **kwargs: Forwarded to :class:`eryn.ensemble.EnsembleSampler`.
    """

    def __init__(self, analysis_container_arr: AnalysisContainerArray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_container_arr = analysis_container_arr

    def get_model(self):
        """Get the ``model`` object the moves receive (overrides eryn's ``Model``).

        Returns:
            GlobalFitInfo: ``(analysis_container_arr, map_fn, random)`` with
            ``map_fn`` the pool's ``map`` (builtin ``map`` without a pool) and
            ``random`` the sampler's RNG.
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
