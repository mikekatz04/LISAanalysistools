"""Top-level runner class (``GlobalFit``).

This module wires together the per-source samplers, the recipe driver, the
HDF backend, and MPI rank coordination into a runnable end-to-end global fit.

MPI rank roles (see ``GlobalFit.resolve_rank_roles``): the main rank runs the
sampler; at ``np >= 3`` the highest spare rank becomes a dedicated
results/saver rank; all other ranks are spares and are stopped at startup.
The legacy multi-stage pipeline classes (``GlobalFitSegment``,
``MPIControlGlobalFit``) were removed 2026-07 (parallel-resources plan P0).
"""

import logging
import os
from copy import deepcopy

import numpy as np
from mpi4py import MPI

try:
    import cupy as xp
    _xp_is_cupy = True
except (ModuleNotFoundError, ImportError):
    import numpy as xp
    _xp_is_cupy = False

    logging.getLogger(__name__).info(
        "cupy not found, using numpy instead. This will be very slow for large runs. "
        "Please install cupy and a compatible CUDA version for GPU acceleration."
    )

from logging import getLogger
import typing

from eryn.state import BranchSupplemental
from eryn.state import State as eryn_State
from eryn.utils.plot import PlotContainer

from ..analysiscontainer import AnalysisContainer, AnalysisContainerArray
from .engine import EngineInfo, GeneralSetup, GlobalFitEngine, GlobalFitSettings, Setup
from .hdfbackend import GFHDFBackend, save_to_backend_asynchronously_and_plot
from .loginfo import dump_settings, init_logger, setup_root_file_handler
from .moves import GFCombineMove, GlobalFitMove, MoveBuildContext
from .postprocessing import GlobalFitPlotter, RunMetadata, SubmissionWriter, save_residuals
from .recipe import Recipe
from .state import GFState
from .utils import BasicResidualacsLikelihood


logger = getLogger(__name__)

#: Branches with hand-written initialization in :meth:`GlobalFit.load_info`.
#: Anything else is seeded by the metadata-driven generic path there.
_LOAD_INFO_NAMED_BRANCHES = ("psd", "galfor", "sgwb", "mbh", "emri", "sobbh", "gb")
class GlobalFitSetup:
    """The built configuration + live state of a global fit (the "Setup").

    Produced from a :class:`~lisatools.globalfit.engine.GlobalFitSettings` the
    same way each per-module ``*Setup`` is built from its ``*Settings``: the
    (heavy) ``__init__`` deepcopies the settings into ``current_info`` and
    opens the HDF backend, then exposes convenient read-only views over the
    configuration — source information, branch states/backends, rank and GPU
    assignments, engine info. This is the object the :class:`GlobalFit` runner
    consumes; a ``GlobalFitSetup`` does not itself drive the sampler.

    The stock layer builds on this: ``StockGlobalFit`` subclasses
    ``GlobalFitSetup`` and defers the heavy ``__init__`` to ``.build()`` (see
    :mod:`lisatools.globalfit.stock`), mirroring ``GlobalFitSettings ->
    GlobalFitSetup -> GlobalFit(run)``.

    .. note::
       Historically named ``CurrentInfoGlobalFit``; that name remains as a
       backward-compatible alias (see below the class).

    Args:
        settings: GlobalFitSettings object containing all configuration parameters
            for the global fit run.
    """

    def __init__(self, settings: GlobalFitSettings):

        self.settings_dict = settings
        self.current_info = deepcopy(settings)

        backend_path = self.general_info.main_file_path
        self.backend = GFHDFBackend(backend_path)

        check = self.engine_info

        # if os.path.exists(mbh_search_file):
        #     with open(mbh_search_file, "rb") as fp:
        #         mbh_output_point_info = pickle.load(fp)

        #     if "output_points_pruned" in mbh_output_point_info:
        #         self.initialize_mbh_state_from_search(mbh_output_point_info)

        # gmm info
        # TODO: save GMM distributions

    # def initialize_mbh_state_from_search(self, mbh_output_point_info):
    #     output_points_pruned = np.asarray(mbh_output_point_info["output_points_pruned"]).transpose(1, 0, 2)
    #     coords = np.zeros((self.source_info["gb"]["pe_info"]["ntemps"], self.source_info["gb"]["pe_info"]["nwalkers"], output_points_pruned.shape[1], self.source_info["mbh"]["pe_info"]["ndim"]))
    #     assert output_points_pruned.shape[0] >= self.source_info["mbh"]["pe_info"]["nwalkers"]

    #     coords[:] = output_points_pruned[None, :self.source_info["mbh"]["pe_info"]["nwalkers"]]
    #     self.source_info["mbh"]["mbh_init_points"] = coords.copy()

    # def get_data_psd(self, **kwargs):
    #     # self passed here to access all current info
    #     return self.general_info["generate_current_state"](self, **kwargs)

    @property
    def branch_names(self) -> typing.List[str]:
        """List of branch names in the global fit model."""
        _names = list(self.source_info.keys())
        return _names

    @property
    def nleaves_max(self) -> typing.Dict[str, int]:
        """Maximum number of leaves for each branch."""
        _nleaves_max = {name: self.source_info[name].nleaves_max for name in self.branch_names}
        return _nleaves_max

    @property
    def nleaves_min(self) -> typing.Dict[str, int]:
        """Minimum number of leaves for each branch."""
        _nleaves_min = {name: self.source_info[name].nleaves_min for name in self.branch_names}
        return _nleaves_min

    @property
    def ndims(self) -> typing.Dict[str, int]:
        """Number of dimensions for each branch."""
        _ndims = {name: self.source_info[name].ndim for name in self.branch_names}
        return _ndims

    @property
    def branch_states(self) -> typing.Dict[str, eryn_State]:
        """Branch state objects for each branch."""
        _branch_states = {name: self.source_info[name].branch_state for name in self.branch_names}
        return _branch_states

    @property
    def branch_backends(self) -> typing.Dict[str, eryn_State]:
        """Branch backend objects for each branch."""
        _branch_backends = {
            name: self.source_info[name].branch_backend for name in self.branch_names
        }
        return _branch_backends

    @property
    def engine_info(self) -> EngineInfo:
        """EngineInfo object containing branch configuration for the sampler engine."""
        engine_info = EngineInfo(
            branch_names=self.branch_names,
            ndims=self.ndims,
            nleaves_max=self.nleaves_max,
            nleaves_min=self.nleaves_min,
            branch_states=self.branch_states,
            branch_backends=self.branch_backends,
        )
        return engine_info

    @property
    def settings(self):
        """GlobalFitSettings dictionary."""
        return self.settings_dict

    @property
    def all_info(self):
        """Complete current information dictionary."""
        return self.current_info

    @property
    def general_info(self) -> GeneralSetup:
        """GeneralSetup object containing general configuration."""
        return self.current_info.general_info

    @property
    def source_info(self):
        """Source-specific configuration information."""
        return self.current_info.source_info

    @property
    def source_metadata(self) -> dict:
        """Metadata information for all sources."""
        return self.current_info.source_metadata

    @property
    def rank_info(self):
        """MPI rank assignment information."""
        return self.current_info.rank_info

    @property
    def gpu_assignments(self):
        """GPU assignment information."""
        return self.current_info.general_info.gpu_assignments

    def get_truths_dict(self) -> dict:
        """Collect injection truths from all source setups for PlotContainer."""
        return {
            name: setup.injection
            for name, setup in self.source_info.items()
            if hasattr(setup, "injection") and setup.injection is not None
        }

    def summarize_run(self, label: str = None, temp: int = 0) -> "GFHDFBackend":
        """Print a compact readout of this fit's *sampled* run and return the reader.

        Reads the fit's own HDF backend, so it works on anything built — a
        stock fit after ``.run()``, or a :class:`GlobalFitSetup` reopened on an
        existing output file. Reports the log-likelihood shape and the final
        value on the ``temp``-th (default cold) chain, then, per branch, the
        chain shape and how many leaves are alive in the last cold-chain
        sample (RJ branches vary; fixed-dimension branches report ``-``).

        Use :meth:`~lisatools.globalfit.stock.base.StockGlobalFit.describe` for
        the *configuration* (and, once built, the resolved per-branch
        products); this is the complement for what the sampler produced.

        Args:
            label: Optional heading, e.g. the stock option name.
            temp: Temperature index to report (0 = the cold chain).

        Returns:
            The backend reader, for further ``get_chain``/``get_inds`` calls.
        """
        reader = self.backend
        ll = reader.get_log_like()
        chain, inds = reader.get_chain(), reader.get_inds()
        print(f"=== {label or type(self).__name__} (sampled) ===")
        print("branches   :", self.branch_names)
        print(f"log_like   : {ll.shape}  final chain (temp {temp}):", np.round(ll[-1, temp], 2))
        for name in self.branch_names:
            alive = int(inds[name][-1, temp].sum()) if name in inds else "-"
            print(
                f"  {name:7s} chain {str(chain[name].shape):26s} "
                f"alive-leaves(temp {temp},last)={alive}"
            )
        return reader


def _periodic_names_to_indices(per_dict: dict, transform) -> dict:
    """Translate a per-branch periodic dict to integer parameter indices.

    Settings files key ``periodic`` by the same parameter names as the
    priors/transform (e.g. ``{"phi0": 2*np.pi}``); eryn's
    :class:`PeriodicContainer` wants sampling-basis integer indices. String
    keys are resolved through ``transform.input_basis``; integer keys pass
    through unchanged.
    """
    out = {}
    basis = getattr(transform, "input_basis", None)
    for var, period in per_dict.items():
        if isinstance(var, str):
            if basis is None or var not in basis:
                raise ValueError(
                    f"periodic parameter {var!r} cannot be resolved to a "
                    "sampling-basis index: the branch transform's "
                    f"input_basis is {basis!r}."
                )
            out[basis.index(var)] = period
        else:
            out[int(var)] = period
    return out


#: Backward-compatible alias for :class:`GlobalFitSetup` (its former name).
#: Kept so existing imports (``from lisatools.globalfit.run import
#: CurrentInfoGlobalFit``) and the legacy ``global_fit_input`` / ``mojito_input``
#: settings files keep working unchanged.
CurrentInfoGlobalFit = GlobalFitSetup


class GlobalFit:
    """The global-fit RUNNER: builds and drives the MCMC sampling run.

    Where :class:`GlobalFitSetup` holds the built configuration + state, this
    class executes it — coordinating MPI rank roles (see
    :meth:`resolve_rank_roles`), GPU assignments, logging, and the MCMC
    workflow that fits multiple gravitational-wave source classes jointly. It
    is composition, not inheritance: a ``GlobalFit`` is constructed with a
    ``GlobalFitSetup`` (``self.curr``) and reads all configuration through it.

    Args:
        curr: GlobalFitSetup object containing all run configuration.
        comm: MPI communicator for parallel processing.
    """

    @classmethod
    def resolve_rank_roles(cls, comm: MPI.Comm, main_rank: int = 0):
        """Resolve the MPI rank roles for a run on ``comm``.

        The layout is: ``main_rank`` runs the sampler; at ``size >= 3`` the
        highest spare rank becomes the dedicated results/saver rank (below
        that, saving is synchronous on main); every remaining rank is a
        spare that is sent ``"stop"`` at startup.

        Exposed as a classmethod so launchers (``scripts/run_global.py``)
        can decide which ranks need the heavy data build *before*
        constructing :class:`GlobalFit`.

        Args:
            comm: MPI communicator.
            main_rank: Rank that drives the sampler. Default 0.

        Returns:
            ``(main_rank, results_rank, spare_ranks)``.
        """
        all_ranks = list(range(comm.Get_size()))
        spares = [r for r in all_ranks if r != main_rank]
        if comm.Get_size() < 3:
            results_rank = main_rank
        else:
            results_rank = spares.pop()
        return main_rank, results_rank, spares

    def __init__(self, curr: GlobalFitSetup, comm: typing.Optional[MPI.Comm] = None):
        """Main class for managing the global fit MCMC sampling run.

        Coordinates MPI processes, GPU assignments, and the MCMC sampling workflow
        for fitting multiple gravitational wave sources simultaneously.

        Args:
            curr: GlobalFitSetup object containing all run configuration.
            comm: MPI communicator for parallel processing. ``None`` (the
                single-process mode used by :meth:`sample`) resolves to
                ``MPI.COMM_SELF``: this rank is main and saver in one, no
                spares, synchronous backend saving.
        """

        self.comm = comm if comm is not None else MPI.COMM_SELF
        self.curr = curr
        self.rank = self.comm.Get_rank()
        self.nwalkers: int = self.curr.general_info.nwalkers
        self.ntemps: int = self.curr.general_info.ntemps
        self.all_ranks = list(range(self.comm.Get_size()))
        # head_rank is a legacy alias from the retired multi-stage pipeline;
        # it gets NO role here. Only ranks with an actual role enter
        # used_ranks — every other rank (including a legacy distinct head
        # rank) is a spare and receives "stop" at startup. Putting a
        # roleless rank in used_ranks deadlocks it in the worker recv.
        self.head_rank = self.curr.rank_info.head_rank
        self.main_rank = self.curr.rank_info.main_rank
        self.main_rank, self.results_rank, self.ranks_to_give = self.resolve_rank_roles(
            self.comm, self.main_rank
        )
        self.used_ranks = [self.main_rank]
        if self.results_rank != self.main_rank:
            self.used_ranks.append(self.results_rank)

        level = logging.DEBUG
        name = "GlobalFit"
        # Console verbosity (general_info.verbose / VERBOSE env / the stock
        # fits' headline knob): quiet default — everything still goes to the
        # run's log files, only warnings/errors reach the console.
        self.verbose = bool(getattr(self.curr.general_info, "verbose", False))
        artifacts_dir = self.curr.general_info.artifacts_file_dir
        setup_root_file_handler(artifacts_dir, level=level)
        self.logger = init_logger(
            filename="global_fit.log", level=level, name=name, log_dir=artifacts_dir,
            console=self.verbose,
        )

        if self.rank == self.main_rank:
            dump_settings(self.curr.settings_dict, artifacts_dir)

    @property
    def _plot_iterations(self) -> int:
        return int(getattr(self.curr.general_info, "plot_iterations", 100) or 100)

    def make_plot_container(self):
        """Build the diagnostic ``PlotContainer`` (or ``None`` when disabled).

        Used by whichever rank owns plotting: the main rank at np < 3, the
        dedicated results rank at np >= 3 (parallel-resources plan P2).
        Opt-out via ``make_diagnostic_plots`` (MAKE_DIAGNOSTIC_PLOTS env on the stock
        classes); cadence via ``plot_iterations`` (PLOT_ITERATIONS env).
        """
        if not getattr(self.curr.general_info, "make_diagnostic_plots", True):
            return None
        branch_names = self.engine_info.branch_names
        truths = self.curr.get_truths_dict()
        exclude_from_plot = ["gb"]  # TODO: make this more general
        truths_plot = {
            key: val for key, val in truths.items() if key not in exclude_from_plot
        }
        branches_plot = [
            name for name in branch_names if name not in exclude_from_plot
        ]
        return PlotContainer(
            plots=["base", "tempering"],
            branches=branches_plot,
            parent_folder=self.curr.general_info.artifacts_file_dir + "diagnostics/",
            tempering_palette="icefire",
            discard=0.3,
            truths=truths_plot,
        )

    def load_info(self, priors: typing.Dict[str, typing.Any]) -> GFState:
        """
        Load or initialize the MCMC state from backend or priors.

        Attempts to load the state from the main backend file. If that doesn't exist,
        tries to load from a past file if specified. Otherwise, initializes a new state
        by drawing from the prior distributions.

        Args:
            priors: Dictionary of prior distributions for each branch.

        Returns:
            GFState object containing the initial or loaded MCMC state.
        """
        self.logger.debug("need to adjust file path")
        # TODO: update to generalize
        state = None
        backend_path = self.curr.general_info.main_file_path
        if os.path.exists(backend_path):
            backend = GFHDFBackend(
                backend_path,
                sub_state_bases=self.engine_info.branch_states,
                sub_backend=self.engine_info.branch_backends,
            )
            # Only load if the backend has been initialized AND has at least
            # one stored sample. Otherwise fall through to past-file or prior
            # initialization. (An empty file gets created when the run sets
            # up artifacts, before the first save_step.)
            if getattr(backend, "initialized", False) and getattr(backend, "iteration", 0) > 0:
                state = backend.get_last_sample()  # .get_a_sample(0)

        if state is None and self.curr.general_info.past_file_for_start is not None:
            # THIS DOES A DIRECT RESTART FROM AN OLD FILE, NO STATISTICAL GENERATION
            if not os.path.exists((file_for_restart := self.curr.general_info.past_file_for_start)):
                raise ValueError(
                    f"past_file_for_start ({file_for_restart}) was added but it does not exist."
                )

            # TODO: make this adjust to more leaves if needed
            state = GFHDFBackend(
                file_for_restart,
                sub_state_bases=self.engine_info.branch_states,
                sub_backend=self.engine_info.branch_backends,
            ).get_last_sample()  # .get_a_sample(0)

            # TODO: adjust this so it is automated
            state.sub_states["gb"].initialized = False
            band_temps = np.zeros((len(self.curr.source_info["gb"].band_edges) - 1, self.ntemps))
            state.sub_states["gb"].initialize_band_information(
                self.nwalkers,
                self.ntemps,
                self.curr.source_info["gb"].band_edges,
                band_temps,
            )

        if state is None:
            self.logger.debug("update this somehow")
            # print("update this somehow")
            # # breakpoint()
            # start from priors by default
            coords = {
                key: priors[key].rvs(
                    size=(self.ntemps, self.nwalkers, self.engine_info.nleaves_max[key])
                )
                for key in self.engine_info.branch_names
            }
            inds = {
                key: np.zeros(
                    (self.ntemps, self.nwalkers, self.engine_info.nleaves_max[key]),
                    dtype=bool,
                )
                for key in self.engine_info.branch_names
            }
            # TODO: make this more generic to anything
            # TODO: this per-branch ``inds[...][:] = True`` flip structure is
            # hand-enumerated branch-by-branch and does not scale — refactor it
            # to drive off branch metadata (e.g. an "always-on"/fixed-leaf flag
            # on the branch settings) instead of a literal if-ladder.
            if "psd" in inds:
                inds["psd"][:] = True
            if "galfor" in inds:
                inds["galfor"][:] = True
            if "sgwb" in inds:
                inds["sgwb"][:] = True
            if "mbh" in inds:
                inds["mbh"][:] = True
                self.logger.debug("initializing mbh inds to true")
                if (
                    "mbh" in self.curr.source_info
                    and self.curr.source_info["mbh"].injection is not None
                ):
                    self.logger.debug(
                        "override mbh starting coords to be close to the injection"
                    )
                    # Starting-point scatter about the injection, env-adjustable:
                    # MBH_START_FACTOR=0 -> exact injection (as the mojito null
                    # checks use); larger -> push the starts further out.
                    factor = float(os.environ.get("MBH_START_FACTOR", "1e-5"))
                    inj = np.asarray(self.curr.source_info["mbh"].injection)
                    if inj.ndim == 1:
                        inj = inj[None, :]
                    nleaves_mbh = self.engine_info.nleaves_max["mbh"]
                    ndim_mbh = inj.shape[-1]
                    if inj.shape[0] == 1:
                        inj = np.broadcast_to(inj, (nleaves_mbh, ndim_mbh))
                    assert inj.shape == (nleaves_mbh, ndim_mbh), (
                        f"MBH injection shape {inj.shape} doesn't match "
                        f"(nleaves_max={nleaves_mbh}, ndim={ndim_mbh})."
                    )
                    coords["mbh"] = inj[None, None] + factor * np.random.randn(
                        self.ntemps, self.nwalkers, nleaves_mbh, ndim_mbh
                    ) 

            if "emri" in inds:
                inds["emri"][:] = True
                self.logger.debug("initializing emri inds to true")

                self.logger.debug("override emri starting coords to be close to the injection")
                # Env-adjustable (EMRI_START_FACTOR=0 -> exact injection).
                factor = float(os.environ.get("EMRI_START_FACTOR", "1e-5"))

                # Multi-leaf safe: accepts either a flat ``(ndim,)`` injection
                # (broadcast across all leaves) or a per-leaf ``(nleaves, ndim)``
                # injection. The trailing axis is always ``ndim`` so the
                # randn matches the engine's ``(ntemps, nwalkers, nleaves, ndim)``
                # coord layout.
                inj = np.asarray(self.curr.source_info["emri"].injection)
                if inj.ndim == 1:
                    inj = inj[None, :]
                nleaves_emri = self.engine_info.nleaves_max["emri"]
                ndim_emri = inj.shape[-1]
                if inj.shape[0] == 1:
                    inj = np.broadcast_to(inj, (nleaves_emri, ndim_emri))
                assert inj.shape == (nleaves_emri, ndim_emri), (
                    f"EMRI injection shape {inj.shape} doesn't match "
                    f"(nleaves_max={nleaves_emri}, ndim={ndim_emri})."
                )
                coords["emri"] = inj[None, None] + factor * np.random.randn(
                    self.ntemps, self.nwalkers, nleaves_emri, ndim_emri
                )
            if "sobbh" in inds:
                inds["sobbh"][:] = True
                self.logger.debug("initializing sobbh inds to true")
                if (
                    "sobbh" in self.curr.source_info
                    and self.curr.source_info["sobbh"].injection is not None
                ):
                    self.logger.debug(
                        "override sobbh starting coords to be close to the injection"
                    )
                    # Env-adjustable (SOBBH_START_FACTOR=0 -> exact injection).
                    factor = float(os.environ.get("SOBBH_START_FACTOR", "1e-5"))
                    inj = np.asarray(self.curr.source_info["sobbh"].injection)
                    if inj.ndim == 1:
                        inj = inj[None, :]
                    nleaves_sobbh = self.engine_info.nleaves_max["sobbh"]
                    ndim_sobbh = inj.shape[-1]
                    if inj.shape[0] == 1:
                        inj = np.broadcast_to(inj, (nleaves_sobbh, ndim_sobbh))
                    assert inj.shape == (nleaves_sobbh, ndim_sobbh), (
                        f"SOBBH injection shape {inj.shape} doesn't match "
                        f"(nleaves_max={nleaves_sobbh}, ndim={ndim_sobbh})."
                    )
                    coords["sobbh"] = inj[None, None] + factor * np.random.randn(
                        self.ntemps, self.nwalkers, nleaves_sobbh, ndim_sobbh
                    )

            # Generic path for any branch the ladder above does not name — a
            # user-added source class. Driven off branch metadata instead of a
            # literal name (see the TODO above): a fixed-leaf branch
            # (nleaves_min == nleaves_max) is always on, and a branch declaring
            # an ``injection`` (sampling basis) starts there, scattered by
            # ``<BRANCH>_START_FACTOR`` — 0 gives the exact injection. Same
            # convention as mbh/emri/sobbh. Without this, a new branch's leaves
            # stay dead, so ``setup_acs`` never subtracts its template and the
            # log-like never returns to ~0 at truth.
            for key in self.engine_info.branch_names:
                if key in _LOAD_INFO_NAMED_BRANCHES or key not in inds:
                    continue
                nleaves_max_key = self.engine_info.nleaves_max[key]
                if self.engine_info.nleaves_min.get(key) == nleaves_max_key:
                    inds[key][:] = True
                    self.logger.debug(f"initializing {key} inds to true (fixed-leaf branch)")
                inj = getattr(self.curr.source_info.get(key), "injection", None)
                if inj is None:
                    continue
                factor = float(os.environ.get(f"{key.upper()}_START_FACTOR", "1e-5"))
                inj = np.asarray(inj, dtype=float)
                if inj.ndim == 1:
                    inj = inj[None, :]
                ndim_key = inj.shape[-1]
                if inj.shape[0] == 1:
                    inj = np.broadcast_to(inj, (nleaves_max_key, ndim_key))
                assert inj.shape == (nleaves_max_key, ndim_key), (
                    f"{key} injection shape {inj.shape} doesn't match "
                    f"(nleaves_max={nleaves_max_key}, ndim={ndim_key})."
                )
                self.logger.debug(f"override {key} starting coords to be close to the injection")
                coords[key] = inj[None, None] + factor * np.random.randn(
                    self.ntemps, self.nwalkers, nleaves_max_key, ndim_key
                )

            state = GFState(
                coords,
                inds=inds,
                random_state=np.random.get_state(),
                sub_state_bases=self.engine_info.branch_states,
            )

            # TODO: generalize all this stuff here (?)
            if "gb" in inds:
                band_temps = np.zeros(
                    (len(self.curr.source_info["gb"].band_edges) - 1, self.ntemps)
                )
                state.sub_states["gb"].initialize_band_information(
                    self.nwalkers,
                    self.ntemps,
                    self.curr.source_info["gb"].band_edges,
                    band_temps,
                )

            state.log_like = np.zeros((self.ntemps, self.nwalkers))
            state.log_prior = np.zeros((self.ntemps, self.nwalkers))
            # self.logger.debug("pickle state load success")

        return state

    def setup_acs(self, state: GFState, rebuild_residuals: bool = False) -> AnalysisContainerArray:
        """
        Set up AnalysisContainerArray for likelihood computations.

        Creates analysis containers for each walker, initializing data
        residuals and sensitivity curves. Domain dispatch (FD / STFT / WDM)
        flows through ``general_info.input_data_residual_array`` and the
        configured :class:`XYZSensitivityBackend`, so nothing in this
        method is FD-specific.

        Args:
            state: GFState object containing current parameter values.
            rebuild_residuals: If ``True``, subtract each non-PSD branch's
                current templates from the freshly-built containers so the
                stored arrays are residuals rather than raw data (stft_tof
                restart/handover path; it was disabled there while the EMRI
                branch was being debugged, so it stays opt-in here).

        Returns:
            AnalysisContainerArray containing data, residuals, and
            sensitivity for all walkers.
        """
        general_info = self.curr.general_info
        if _xp_is_cupy and general_info.gpus is not None:
            xp.cuda.runtime.setDevice(general_info.gpus[0])

        # Per-branch params-based template generators registered into every
        # AC's dictionary-based ``signal_gen``. Settings expose them as
        # ``source_info[name].signal_gen`` (the converted core of the legacy
        # ``get_templates`` process: transform + waveform generator called as
        # ``fn(*params) -> template``). Branches without one fall back to the
        # bulk ``get_templates`` hook in the rebuild loop below until they
        # are converted.
        signal_gen_map = {}
        for name in self.curr.engine_info.branch_names:
            if name in ("psd", "galfor") or name not in self.curr.source_info:
                continue
            _gen = getattr(self.curr.source_info[name], "signal_gen", None)
            if callable(_gen):
                signal_gen_map[name] = _gen

        # Run-level likelihood convention (see GeneralSettings docstring).
        # getattr for backward compatibility with pickled/legacy settings.
        ll_source_only = bool(getattr(general_info, "likelihood_source_only", False))
        if ll_source_only and "psd" in self.curr.engine_info.branch_names:
            logger.warning(
                "likelihood_source_only=True with a 'psd' sampling branch: the "
                "noise term varies with the PSD parameters, so source-only "
                "likelihoods are NOT valid for PSD acceptance. Disabling."
            )
            ll_source_only = False

        acs_tmp = []
        for w in range(self.nwalkers):
            data_res_arr = deepcopy(general_info.input_data_residual_array)
            if "psd" in state.branches_coords.keys():
                psd_params = state.branches_coords["psd"][0, w, 0]
                psd_params = (
                    self.curr.source_info["psd"].transform.both_transforms(psd_params)
                    if self.curr.source_info["psd"].transform is not None
                    else psd_params
                )
                # need to generalize for other stochastic functions
                if "galfor" in state.branches_coords.keys():
                    galfor_params = state.branches_coords["galfor"][0, w, 0]
                    galfor_params = (
                        self.curr.source_info["galfor"].transform.both_transforms(galfor_params)
                        if self.curr.source_info["galfor"].transform is not None
                        else galfor_params
                    )
                else:
                    galfor_params = None
                # only forward sgwb_params when the branch exists so the legacy
                # XYZSensitivityBackend signature keeps working for runs without
                # an sgwb branch
                extra_sens_kwargs = {}
                if "sgwb" in state.branches_coords.keys():
                    sgwb_params = state.branches_coords["sgwb"][0, w, 0]
                    sgwb_params = (
                        self.curr.source_info["sgwb"].transform.both_transforms(sgwb_params)
                        if self.curr.source_info["sgwb"].transform is not None
                        else sgwb_params
                    )
                    extra_sens_kwargs["sgwb_params"] = sgwb_params
                sens_here = general_info.sensitivity_backend(
                    f"walker_{w}",
                    psd_params,
                    transform_fn=self.curr.source_info["psd"].transform_fn,
                    galfor_params=galfor_params,
                    **extra_sens_kwargs,
                )
            else:
                sens_here = general_info.sensitivity_backend(
                    f"walker_{w}", **general_info.fixed_psd_kwargs
                )

            acs_tmp.append(
                (
                    _analysis_tmp := AnalysisContainer(
                        deepcopy(data_res_arr),
                        deepcopy(sens_here),
                        signal_gen=dict(signal_gen_map) if signal_gen_map else None,
                        likelihood_source_only=ll_source_only,
                    )
                )
            )

        gpus = general_info.gpus
        acs = AnalysisContainerArray(acs_tmp, gpus=gpus)

        if rebuild_residuals:
            # Residual rebuild, replicating the stft_tof ``get_templates``
            # process. Preferred route (2026-06 merge direction): drive the
            # template generation from the state's coords/inds through each
            # container's dictionary-based ``signal_gen``
            # ({branch_name -> generator}) and
            # :meth:`AnalysisContainer.build_template` -- no model callables
            # passed through ``source_info``. Branches whose generators are
            # not (yet) registered on ``signal_gen`` fall back to the
            # stft_tof ``source_info[...]["get_templates"]`` process so the
            # rebuild always works during the migration.
            # A branch with a registered generator is handled by this path even
            # when no leaf is currently alive (nothing to subtract) — seeding
            # this from the map rather than from the alive-leaf params below
            # keeps the fallback loop from warning about branches that are in
            # fact correctly configured.
            handled_by_signal_gen = set(signal_gen_map)
            for w, ac in enumerate(acs.flatten()):
                gen_map = getattr(ac, "_signal_gen", None)
                if not isinstance(gen_map, dict):
                    continue  # this walker's branches use the fallback below
                params = {}
                for name in self.curr.engine_info.branch_names:
                    if name in ("psd", "galfor") or name not in gen_map:
                        continue
                    inds_w = state.branches_inds[name][0, w]
                    if not inds_w.any():
                        continue
                    params[name] = state.branches_coords[name][0, w][inds_w]
                handled_by_signal_gen.update(params.keys())
                if not params:
                    continue
                template = ac.build_template(params)
                # breakpoint()  # debug hook: inspect template vs ac.data here
                ac.data.add_signal(template, sign=-1)

            # stft_tof fallback for branches without a registered generator.
            for name, source_info in self.curr.source_info.items():
                if name not in self.curr.engine_info.branch_names:
                    continue
                if name in ("psd", "galfor") or name in handled_by_signal_gen:
                    continue
                try:
                    get_templates = source_info["get_templates"]
                except (KeyError, TypeError):
                    logger.warning(
                        f"rebuild_residuals: branch {name!r} has neither a "
                        "signal_gen entry nor a get_templates hook; skipped."
                    )
                    continue

                templates_tmp = xp.asarray(
                    get_templates(state, source_info, self.curr.general_info)
                )

                # no need to adjust data index or start_freq_ind:
                # add_signal_to_residual handles alignment.
                acs.add_signal_to_residual(templates_tmp)

                del templates_tmp
                logger.info(f"added {name} templates to acs residuals (get_templates path).")

            if xp.__name__ == "cupy":
                xp.get_default_memory_pool().free_all_blocks()
            logger.info("rebuilt residuals from state coords/inds.")

        return acs

    @property
    def engine_info(self) -> EngineInfo:
        """EngineInfo object containing branch configuration for the sampler engine."""
        return self.curr.engine_info

    def prepare_main(self):
        """Build everything the sampling rank needs: backend, state, ACS, engine, recipe.

        Extracted from ``run_global_fit`` (identical behavior): opens the HDF
        backend, loads/initializes the state, builds the shared analysis
        containers and likelihood, constructs the :class:`GlobalFitEngine`,
        invokes the ``setup_function`` (which materializes the recipe), and
        wires the recipe/backend bookkeeping. Afterwards ``self.sampler`` /
        ``self.state`` / ``self.priors`` / ``self.acs`` / ``self.run_backend``
        / ``self.live_ctx`` are set; ``run_global_fit`` and :meth:`sample`
        both start from here.
        """
        backend_path = self.curr.general_info.main_file_path

        general_info = self.curr.general_info

        branch_names = self.engine_info.branch_names
        ndims = self.engine_info.ndims
        nleaves_max = self.engine_info.nleaves_max
        nleaves_min = self.engine_info.nleaves_min
        nwalkers = general_info.nwalkers
        ntemps = general_info.ntemps

        priors = {}
        periodic = {}
        for name in branch_names:
            # TODO: clean up, but also inform using current_info: Settings? = self.curr.source_info[name]
            if name not in self.curr.source_info:
                continue

            if isinstance(self.curr.source_info[name], dict):
                for key, value in self.curr.source_info[name]["priors"].items():
                    priors[key] = value

                if (
                    "periodic" in self.curr.source_info[name]
                    and self.curr.source_info[name]["periodic"] is not None
                ):
                    for key, value in self.curr.source_info[name]["periodic"].items():
                        periodic[key] = _periodic_names_to_indices(
                            value, self.curr.source_info[name].get("transform")
                        )

            # TODO: clean up
            if isinstance(self.curr.source_info[name], Setup):
                for key, value in self.curr.source_info[name].priors.items():
                    priors[key] = value

                if (
                    hasattr(self.curr.source_info[name], "periodic")
                    and self.curr.source_info[name].periodic is not None
                ):
                    for key, value in self.curr.source_info[name].periodic.items():
                        periodic[key] = _periodic_names_to_indices(
                            value, getattr(self.curr.source_info[name], "transform", None)
                        )

        state = self.load_info(priors)
        self.logger.debug("state loaded")

        supps_base_shape = (ntemps, nwalkers)
        walker_vals = np.tile(np.arange(nwalkers), (ntemps, 1))
        supps = BranchSupplemental(
            {"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True
        )
        state.supplemental = supps
        # breakpoint()

        # backend.reset(
        #     nwalkers,
        #     ndims,
        #     nleaves_max=nleaves_max,
        #     ntemps=ntemps,
        #     branch_names=branch_names,
        #     nbranches=len(branch_names),
        #     rj=True,
        #     moves=None,
        #     num_mbhs=nleaves_max["mbh"],
        #     num_bands=state.sub_states["gb"].band_info["num_bands"],
        #     band_edges=state.sub_states["gb"].band_info["band_edges"],
        # )

        # backend.grow(1, None)

        # gb_backend = HDFBackend("global_fit_output/eighth_run_through_parameter_estimation_gb.h5")
        # psd_backend = HDFBackend("global_fit_output/eighth_run_through_parameter_estimation_psd.h5")
        # mbh_backend = HDFBackend("global_fit_output/eighth_run_through_parameter_estimation_mbh.h5")

        # last_gb = gb_backend.get_last_sample()
        # last_psd = psd_backend.get_last_sample()
        # last_mbh = mbh_backend.get_last_sample()

        # state.branches["gb"] = deepcopy(last_gb.branches["gb"])
        # state.branches["psd"].coords[:] = last_psd.branches["psd"].coords[0, :nwalkers]
        # # order of call function changed for galfor
        # galfor_coords_orig = last_psd.branches["galfor"].coords[0, :nwalkers]
        # galfor_coords = np.zeros_like(galfor_coords_orig)
        # galfor_coords[:, :, 0] = galfor_coords_orig[:, :, 0]
        # galfor_coords[:, :, 1] = galfor_coords_orig[:, :, 3]
        # galfor_coords[:, :, 2] = galfor_coords_orig[:, :, 1]
        # galfor_coords[:, :, 3] = galfor_coords_orig[:, :, 2]
        # galfor_coords[:, :, 4] = galfor_coords_orig[:, :, 4]
        # state.branches["galfor"].coords[:] = galfor_coords
        # state.branches["mbh"].coords[:] = last_mbh.branches["mbh"].coords[0, :nwalkers]

        # # FOR TESTING
        # state.branches["gb"].coords[:] = state.branches["gb"].coords[0, 0][None, None, :, :]
        # state.branches["gb"].inds[:] = state.branches["gb"].inds[0, 0][None, None, :]
        # state.branches["mbh"].coords[:] = state.branches["mbh"].coords[0, 0][None, None, :, :]
        # state.branches["psd"].coords[:] = state.branches["psd"].coords[0, 0][None, None, :, :]
        # state.branches["galfor"].coords[:] = state.branches["galfor"].coords[0, 0][None, None, :, :]

        # accepted = np.zeros((ntemps, nwalkers), dtype=int)
        # swaps_accepted = np.zeros((ntemps - 1,), dtype=int)
        # state.log_like = np.zeros((ntemps, nwalkers))
        # state.log_prior = np.zeros((ntemps, nwalkers))
        # state.betas = np.ones((ntemps,))

        # backend.save_step(state, accepted, rj_accepted=accepted, swaps_accepted=swaps_accepted)

        # A_inj = general_info.A_inj.copy()
        # E_inj = general_info.E_inj.copy()

        # generate = GenerateCurrentState(A_inj, E_inj)
        # self.logger.debug("generate function created")

        # rebuild_residuals=True: branches that registered a params-based
        # ``signal_gen`` on their Setup get their current templates
        # subtracted here, under the hood (the converted ``get_templates``
        # process). Branches without one are skipped with a warning and
        # may keep subtracting in their recipe (legacy path) -- no
        # double-subtraction either way.
        acs = self.setup_acs(state, rebuild_residuals=True)
        self.logger.debug("acs setup done")

        state.log_like[:] = acs.likelihood(complex=False)
        logger.info(f"initial log likelihood: {state.log_like[0]}")

        like_mix = BasicResidualacsLikelihood(acs)

        backend = GFHDFBackend(
            backend_path,  # self.curr.general_info["file_information"]["fp_main"],
            # gzip-4 default: level 9 was CPU-heavy on the saver for
            # marginal size gains on chain data (settings-overridable).
            compression=getattr(
                self.curr.general_info, "hdf_compression", "gzip"
            ),
            compression_opts=getattr(
                self.curr.general_info, "hdf_compression_opts", 4
            ),
            comm=self.comm,
            save_plot_rank=self.results_rank,
            sub_backend=self.engine_info.branch_backends,
            sub_state_bases=self.engine_info.branch_states,
        )

        extra_reset_kwargs = {}
        # TODO: fix this somehow
        for name in branch_names:
            if name in state.sub_states and state.sub_states[name] is not None:
                extra_reset_kwargs = {
                    **extra_reset_kwargs,
                    **state.sub_states[name].reset_kwargs,
                }

        if not backend.initialized:
            # ``key_order`` mirrors what eryn's EnsembleSampler would
            # pass to ``backend.reset`` itself when the backend is fresh;
            # we have to feed it in here because our pre-reset disables
            # that branch (``self.backend.initialized`` becomes True
            # before the sampler is constructed). Without it, the
            # sampler's later ``self.key_order != self.backend.key_order``
            # check fires.
            key_order = {
                key: value.key_order for key, value in priors.items()
            }
            backend.reset(
                nwalkers,
                ndims,
                nleaves_max=nleaves_max,
                ntemps=ntemps,
                branch_names=branch_names,
                nbranches=len(branch_names),
                rj=False,
                moves=None,
                key_order=key_order,
                **extra_reset_kwargs,
            )

            # Persist the domain settings (FD / STFT / WDM) and the
            # optional WDM lookup table so a re-run can reconstruct
            # everything from a single HDF5 file. ``general_info``
            # already holds the resolved instances at this point.
            domain_settings = general_info.domain_settings
            if domain_settings is not None:
                backend.write_domain_settings(domain_settings)
            if getattr(general_info, "wdm_lookup_table", None) is not None:
                backend.write_wdm_lookup_table(general_info.wdm_lookup_table)

        # setup_info_all = None
        # for name in branch_names:
        #     if name not in self.curr.source_info:
        #         setup_info = SetupInfoTransfer(name=name)

        #     elif "setup_func" in self.curr.source_info[name]:
        #         setup_info = self.curr.source_info[name]["setup_func"](self.gf_branch_information, self.curr, acs, priors, state)
        #     else:
        #         setup_info = SetupInfoTransfer(name=name)

        #     if setup_info_all is None:
        #         setup_info_all = setup_info
        #     else:
        #         setup_info_all += setup_info

        # The configured fit's recipe rides in source_metadata (stock
        # path: fit.recipe IS the object that runs); the setup_function
        # materializes it via recipe.setup(ctx). Legacy settings-file
        # runs get a fresh empty Recipe to fill directly.
        recipe = self.curr.source_metadata.get("recipe")
        if recipe is None:
            recipe = Recipe()
        recipe._init_runtime()
        self.recipe = recipe
        setup_info_all = self.curr.settings_dict.setup_function(
            self.recipe, self.engine_info, self.curr, acs, priors, state
        )

        logger.debug("need to setup moves that use parallel resources")

        # backend.grow(1, None)
        # accepted = np.zeros((self.ntemps, self.nwalkers), dtype=int)
        # swaps_accepted = np.zeros((self.ntemps - 1), dtype=int)
        # backend.save_step(state, accepted, swaps_accepted=swaps_accepted)
        # exit()

        # Stop the spare processes. (The old move->rank dispatch that
        # handed spares to moves was removed with the CPU distribution-
        # fitting workers it served — GPU GMM fitting / neural flows
        # replaced them; parallel-resources plan P3. A future coarse
        # multi-node worker pool would re-enter here.)
        for rank in self.all_ranks:
            if rank in self.used_ranks:
                continue
            self.comm.send("stop", dest=rank)

        from eryn.moves import StretchMove

        _tmp_move = StretchMove(live_dangerously=True)
        # permute False is there for the PSD sampling for now

        # Diagnostic plotting ownership (parallel-resources plan P2): at
        # np >= 3 the dedicated results rank renders the plots from the
        # backend it writes, so the sampler never blocks on matplotlib;
        # below that the main rank plots as before.
        _plot_iterations = self._plot_iterations
        plot_container = (
            self.make_plot_container()
            if self.results_rank == self.main_rank
            else None
        )
        if plot_container is None:
            # eryn auto-creates its own PlotContainer when
            # plot_generator is None and plot_iterations > 0.
            _plot_iterations = -1

        # Wrap ``periodic`` as a ``PeriodicContainer`` with ``key_order``
        # so eryn doesn't reject the string-keyed dict. The key_order
        # for each branch comes from its prior's ``key_order``.
        from eryn.utils import PeriodicContainer

        periodic_key_order = {
            key: value.key_order for key, value in priors.items()
        }
        if periodic and not isinstance(periodic, PeriodicContainer):
            periodic = PeriodicContainer(periodic, key_order=periodic_key_order)

        sampler_mix = GlobalFitEngine(
            acs,
            self.nwalkers,
            ndims,  # assumes ndim_max
            like_mix,
            priors,
            tempering_kwargs={"ntemps": self.ntemps},
            nbranches=len(branch_names),
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=_tmp_move,  # setup_info_all.in_model_moves_input,
            rj_moves=None,  # setup_info_all.rj_moves_input,
            kwargs=None,
            backend=backend,
            vectorize=True,
            periodic=periodic,
            branch_names=branch_names,
            # update_fn=update_fn,
            plot_generator=plot_container,
            plot_iterations=_plot_iterations,
            # update_iterations=1,
            # update_fn=recipe,  # stop_converge_mix,
            # update_iterations=1,  # TODO: change this?
            provide_groups=True,
            provide_supplemental=True,
            track_moves=False,
            stopping_fn=self.recipe,
            stopping_iterations=1,
        )
        _tmp_move.temperature_control.swaps_accepted = np.zeros((self.ntemps - 1), dtype=int)

        self.recipe.backend = backend
        backend.add_recipe(self.recipe)

        # ``sum_instead_of_trapz`` was a legacy ``inner_product`` knob
        # that no longer exists; the modern inner_product already does
        # the sum-style integration by default.
        state.log_like[:] = acs.likelihood(complex=False)[None, :]
        state.log_prior = np.zeros_like(
            state.log_like
        )  # sampler_mix.compute_log_prior(state.branches_coords, inds=state.branches_inds, supps=supps)
        self.recipe.setup_first_recipe_step(sampler_mix.iteration, state, sampler_mix)

        if self.curr.general_info.submission_parent_folder is not None:
            gf_plotter = GlobalFitPlotter(curr=self.curr)
            gf_plotter.save_input_data()

        # Everything sample()/run_global_fit need to proceed, plus the live
        # context that lets add_move materialize into a running fit.
        self.sampler = sampler_mix
        self.state = state
        self.priors = priors
        self.acs = acs
        self.run_backend = backend
        self.live_ctx = MoveBuildContext(
            recipe=self.recipe,
            engine_info=self.engine_info,
            curr=self.curr,
            acs=acs,
            priors=priors,
            state=state,
            stock_moves=getattr(self.recipe, "stock_moves", {}),
            ntemps=self.ntemps,
            nwalkers=self.nwalkers,
        )

    def run_global_fit(self):
        """Execute the main global fit MCMC sampling run.

        Coordinates the entire sampling workflow including:
        - Setting up the backend for storing results (see :meth:`prepare_main`)
        - Loading or initializing the state
        - Setting up analysis containers and likelihood
        - Configuring the sampler with moves and priors
        - Running the MCMC chain
        - Distributing tasks across MPI ranks
        """

        backend_path = self.curr.general_info.main_file_path

        backend = GFHDFBackend(
            backend_path,
            sub_backend=self.engine_info.branch_backends,
            sub_state_bases=self.engine_info.branch_states,
        )
        if self.rank == self.curr.settings_dict.rank_info.main_rank:
            self.prepare_main()

            self.sampler.run_mcmc(self.state, self.curr.general_info.num_iterations, thin_by=1, progress=self.verbose, store=True)

            if self.curr.general_info.submission_parent_folder is not None:
                self.logger.debug(f"saving submission to {self.curr.general_info.submission_parent_folder}")
                submission_writer = SubmissionWriter(backend=self.run_backend, curr=self.curr, ess=20_000)
                submission_writer.write_submission(self.acs)

            logger.info("Residuals saved.")

            if self.results_rank != self.main_rank:
                # Dedicated saver rank (np >= 3): tell it the run is over.
                # Below that the saver is aliased to main — a self-send would
                # only rely on MPI eager buffering for nothing.
                self.comm.send({"finish_run": True}, dest=self.results_rank)

        elif self.rank == self.results_rank:
            # Dedicated saver rank (np >= 3): async HDF5 writes + the
            # diagnostic plots, both off the sampler's critical path
            # (saves always take priority; see the loop's docstring).
            save_to_backend_asynchronously_and_plot(
                backend,
                self.comm,
                self.main_rank,
                plot_container=self.make_plot_container(),
                plot_iter=self._plot_iterations,
                backup_iter=self.curr.general_info.backup_iter,
            )

        else:
            # Spare rank: wait for the startup "stop" and exit. (The
            # instruction-dict dispatch that ran move workers here was
            # removed with the move->rank machinery; plan P3.)
            info = self.comm.recv(source=self.main_rank)
            logger.info(f"Process {self.rank} finished ({info!r}).")

    def sample(
        self,
        iterations: typing.Optional[int] = None,
        *,
        thin_by: int = 1,
        progress: bool = False,
        store: bool = True,
        sync_log_like: bool = True,
    ):
        """Generator run mode: yield ``(model, state)`` once per iteration.

        The emcee-style loop, one level up from the engine's own ``sample``
        (which it wraps)::

            gf = GlobalFit(curr)              # comm=None -> single process
            for model, state in gf.sample(iterations=100):
                ...   # inspect/mutate model.analysis_container_arr and state
                      # in place; the next iteration continues from them

        In-place mutation propagates because the yielded ``state`` is exactly
        the object fed to the next iteration. The recipe's stage-advance logic
        runs here each iteration (under ``run_mcmc`` the stopping function
        owns it), so multi-stage recipes behave identically; the loop ends
        when the recipe finishes or ``iterations`` is exhausted.

        Single-process only (``run_global_fit`` owns MPI): inside the loop you
        may do whatever you need — including your own MPI — as long as control
        returns synchronously.

        .. note::
           The backend saves each step *before* the yield, so an in-loop
           mutation is persisted with the *next* saved step; mutations after
           the final yield are not saved.

        Args:
            iterations: Iterations to run; ``None`` -> the configured
                ``general.num_iterations``.
            thin_by: Yield every ``thin_by``-th iteration (forwarded to the
                engine).
            progress: Show the engine's progress bar.
            store: Save steps to the HDF backend.
            sync_log_like: After each yield, re-sync ``state.log_like`` from
                the residual (``acs.likelihood()``) so in-loop residual
                mutations flow into the tempering/persistence bookkeeping. If
                you mutate ``coords`` in place, update ``state.log_prior``
                yourself.
        """
        if self.comm.Get_size() > 1:
            raise RuntimeError(
                "sample() is single-process; run under MPI with run_global_fit() "
                "(sample() itself may be used inside code that does its own MPI)."
            )
        self.prepare_main()
        sampler, state = self.sampler, self.state
        if iterations is None:
            iterations = self.curr.general_info.num_iterations
        i = 0
        try:
            for state in sampler.sample(
                state, iterations=iterations, thin_by=thin_by, store=store, progress=progress
            ):
                # Recipe stage-advance: run_mcmc drives this via stopping_fn;
                # the engine's sample() generator does not, so drive it here
                # at the same per-iteration cadence.
                if self.recipe(i, state, sampler):
                    break
                yield sampler.get_model(), state
                if sync_log_like:
                    state.log_like[:] = sampler.analysis_container_arr.likelihood(
                        complex=False
                    )[None, :]
                i += 1
        finally:
            # Resumable: a later run_mcmc/sample continues from the last state.
            sampler._previous_state = state
            self.live_ctx = None
