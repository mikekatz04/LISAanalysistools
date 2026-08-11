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

from contextlib import nullcontext as _nullcontext


def _rss_mb() -> float:
    """Current process max-RSS in MB (Linux reports KB, macOS bytes).

    Used by the fresh-start checkpoint logging: a run killed by a cgroup /
    OOM limit dies silently mid-allocation, so each checkpoint stamps the
    high-water mark that was reached before it."""
    import resource
    import sys

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0 if sys.platform.startswith("linux") else ru / (1024.0**2)

from ..analysiscontainer import AnalysisContainer, AnalysisContainerArray
from ..utils.device import device_context, pin_main_device
from ..utils.utility import asnumpy
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
        # Progress bar is a separate knob (general_info.progress / PROGRESS):
        # None follows verbose -- the historical pairing -- so a run that only
        # wants the tqdm bar can have it without DEBUG logs on the console,
        # and a verbose run writing to a log file can suppress the bar.
        _progress = getattr(self.curr.general_info, "progress", None)
        self.progress = self.verbose if _progress is None else bool(_progress)
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
        # the tempering (swap-fraction) plot is meaningless at engine
        # ntemps=1 -- module tempering lives in the sub-backends now
        _plots = ["base", "tempering"] if self.ntemps > 1 else ["base"]
        return PlotContainer(
            plots=_plots,
            branches=branches_plot,
            parent_folder=self.curr.general_info.artifacts_file_dir + "diagnostics/",
            tempering_palette="icefire",
            discard=0.3,
            truths=truths_plot,
        )

    def _branch_ntemps(self, name: str) -> int:
        """The branch's OWN tempering-ladder size (the engine is cold-chain only).

        An explicit per-branch ``betas`` ladder wins and defines its length;
        otherwise the branch's ``ntemps`` setting. Branches with neither
        (e.g. simple-API branches) run on the engine ladder.
        """
        info = self.curr.source_info.get(name)
        if info is None:
            return self.ntemps
        betas = getattr(info, "betas", None)
        if betas is not None:
            return len(betas)
        nt = getattr(info, "ntemps", None)
        return int(nt) if nt else self.ntemps

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
                # clean-break gate: old-layout files (full-ntemps main chain)
                # cannot be resumed; this fires with an actionable message
                # BEFORE eryn's opaque backend-shape check would.
                backend.check_format_version("resume")
                state = backend.get_last_sample()  # .get_a_sample(0)
                self.logger.info(
                    "RESUMING from existing backend %s at stored iteration %d "
                    "(the 'initial log likelihood' below is that state, NOT a "
                    "fresh start; new iterations append).",
                    backend_path, int(backend.iteration),
                )
                # Guard against resuming a backend whose per-branch sampled
                # dimensionality no longer matches the run config -- the most
                # likely cause is toggling GB_USE_ASTROPHYSICAL_F0_MC_PRIOR /
                # GB_USE_CHIRP_MASS (8 <-> 9 column GB basis) between runs.
                for _name, _nd in self.curr.ndims.items():
                    _coords = getattr(state, "branches_coords", {}).get(_name)
                    if _coords is not None and _coords.shape[-1] != _nd:
                        raise ValueError(
                            f"Cannot resume {backend_path!r}: branch {_name!r} "
                            f"stored with ndim {_coords.shape[-1]} but the run "
                            f"config expects {_nd}. For GB this usually means "
                            f"GB_USE_ASTROPHYSICAL_F0_MC_PRIOR / "
                            f"GB_USE_CHIRP_MASS differ from the stored run "
                            f"(8-col vs 9-col fdot_astro_ratio basis). Start a "
                            f"fresh backend or match the original config."
                        )

        if state is None and self.curr.general_info.past_file_for_start is not None:
            # THIS DOES A DIRECT RESTART FROM AN OLD FILE, NO STATISTICAL GENERATION
            if not os.path.exists((file_for_restart := self.curr.general_info.past_file_for_start)):
                raise ValueError(
                    f"past_file_for_start ({file_for_restart}) was added but it does not exist."
                )

            # TODO: make this adjust to more leaves if needed
            _restart_backend = GFHDFBackend(
                file_for_restart,
                sub_state_bases=self.engine_info.branch_states,
                sub_backend=self.engine_info.branch_backends,
            )
            _restart_backend.check_format_version("past_file_for_start")
            state = _restart_backend.get_last_sample()  # .get_a_sample(0)

            # TODO: adjust this so it is automated
            _nt_gb = self._branch_ntemps("gb")
            band_temps = np.zeros((len(self.curr.source_info["gb"].band_edges) - 1, _nt_gb))
            state.sub_states["gb"].initialize_band_information(
                self.nwalkers,
                _nt_gb,
                self.curr.source_info["gb"].band_edges,
                band_temps,
            )

        if state is None:
            self.logger.info(
                "FRESH START: no resumable backend (missing, empty, or "
                "zero stored iterations) and no past_file_for_start — "
                "initializing from priors/injection."
            )
            # start from priors by default. Draw at the WIDEST ladder any
            # branch needs, then slice: the engine keeps only its own ladder
            # (cold chain for stock variants) while each sub-state takes its
            # branch's full ladder. Draws are iid along the temp axis, so
            # per-branch slicing preserves the draw statistics.
            nt_draw = max(
                [self.ntemps]
                + [
                    self._branch_ntemps(key)
                    for key in self.engine_info.branch_names
                    if self.engine_info.branch_states.get(key) is not None
                ]
            )
            # Per-branch checkpoints with the RSS high-water mark: a cgroup /
            # OOM kill in this segment is silent, so the last line that made
            # it to global_fit.log localizes the allocation that died.
            self.logger.info(
                "fresh start: drawing priors at nt_draw=%d nwalkers=%d "
                "(RSS %.0f MB)", nt_draw, self.nwalkers, _rss_mb(),
            )
            coords = {}
            for key in self.engine_info.branch_names:
                shape = (nt_draw, self.nwalkers, self.engine_info.nleaves_max[key])
                self.logger.info(
                    "fresh start: drawing '%s' priors, shape %s (RSS %.0f MB)",
                    key, shape, _rss_mb(),
                )
                coords[key] = priors[key].rvs(size=shape)
            self.logger.info("fresh start: prior draws done (RSS %.0f MB)", _rss_mb())
            inds = {
                key: np.zeros(
                    (nt_draw, self.nwalkers, self.engine_info.nleaves_max[key]),
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
                    # MULTIPLICATIVE ``x * (1 + factor * randn)`` (the sprint-wide
                    # START_FACTOR convention). MBH_START_FACTOR=0 -> exact
                    # injection (as the mojito null checks use); larger -> push
                    # the starts further out.
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
                    coords["mbh"] = inj[None, None] * (
                        1.0
                        + factor
                        * np.random.randn(
                            nt_draw, self.nwalkers, nleaves_mbh, ndim_mbh
                        )
                    )

            if "emri" in inds:
                inds["emri"][:] = True
                self.logger.debug("initializing emri inds to true")

                self.logger.debug("override emri starting coords to be close to the injection")
                # Env-adjustable, MULTIPLICATIVE ``x * (1 + factor * randn)``
                # (EMRI_START_FACTOR=0 -> exact injection).
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
                coords["emri"] = inj[None, None] * (
                    1.0
                    + factor
                    * np.random.randn(
                        nt_draw, self.nwalkers, nleaves_emri, ndim_emri
                    )
                )
            if "gb" in inds and getattr(
                self.curr.source_info.get("gb"), "injection", None
            ) is not None:
                # gb: RJ branch seeded from the attach-time SNR-cut rows
                # (``gb_info.injection``; gb_no_fg resolves them against a
                # noise-only AnalysisContainer). Only the subset's leaves go
                # alive -- gb is NOT a fixed-leaf branch. Scatter follows the
                # sprint-wide START_FACTOR convention: MULTIPLICATIVE
                # ``x * (1 + factor * randn)`` (0 -> exact truth). Seeding
                # here (with everything else) lets setup_acs's engine
                # rebuild subtract the templates through the registered
                # signal_gen in the same pass as every other branch.
                inj = np.asarray(
                    self.curr.source_info["gb"].injection, dtype=float
                )
                n_inj, ndim_gb = inj.shape
                nleaves_gb = self.engine_info.nleaves_max["gb"]
                assert n_inj <= nleaves_gb, (
                    f"GB injection rows ({n_inj}) exceed nleaves_max "
                    f"({nleaves_gb})."
                )
                factor = float(os.environ.get("GB_START_FACTOR", "1e-4"))
                inds["gb"][:] = False
                inds["gb"][:, :, :n_inj] = True
                coords["gb"][:, :, :n_inj, :] = inj[None, None] * (
                    1.0
                    + factor
                    * np.random.randn(nt_draw, self.nwalkers, n_inj, ndim_gb)
                )
                self.logger.info(
                    f"gb: seeded {n_inj} true-point leaves in load_info "
                    f"(GB_START_FACTOR={factor:g}); engine rebuild subtracts "
                    "them with every other branch."
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
                    # Env-adjustable, MULTIPLICATIVE ``x * (1 + factor * randn)``
                    # (SOBBH_START_FACTOR=0 -> exact injection).
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
                    coords["sobbh"] = inj[None, None] * (
                        1.0
                        + factor
                        * np.random.randn(
                            nt_draw, self.nwalkers, nleaves_sobbh, ndim_sobbh
                        )
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
                # MULTIPLICATIVE scatter ``x * (1 + factor * randn)``: each
                # parameter is perturbed by a FRACTION of its own value, so
                # dimensions of wildly different magnitude (e.g. GB/VGB fdot
                # ~1e-16 alongside a ln-amplitude ~-50) all scatter sensibly
                # off the injection without a per-dimension covariance/width
                # scale. ``factor = 0`` -> exact injection (truth-null checks).
                coords[key] = inj[None, None] * (
                    1.0
                    + factor
                    * np.random.randn(
                        nt_draw, self.nwalkers, nleaves_max_key, ndim_key
                    )
                )

            # the main state keeps only the engine's ladder (cold chain for
            # stock variants); each sub-state takes its branch's full ladder
            coords_full, inds_full = coords, inds
            coords = {key: value[: self.ntemps].copy() for key, value in coords_full.items()}
            inds = {key: value[: self.ntemps].copy() for key, value in inds_full.items()}

            self.logger.info("fresh start: building GFState (RSS %.0f MB)", _rss_mb())
            state = GFState(
                coords,
                inds=inds,
                random_state=np.random.get_state(),
                sub_state_bases=self.engine_info.branch_states,
            )

            for key, sub in state.sub_states.items():
                if sub is None:
                    continue
                nt_branch = self._branch_ntemps(key)
                self.logger.info(
                    "fresh start: tempered sub-state '%s' nt=%d (RSS %.0f MB)",
                    key, nt_branch, _rss_mb(),
                )
                sub.initialize_tempered(
                    nt_branch,
                    self.nwalkers,
                    self.engine_info.nleaves_max[key],
                    self.engine_info.ndims[key],
                    coords=coords_full[key][:nt_branch],
                    inds=inds_full[key][:nt_branch],
                )

            # TODO: generalize all this stuff here (?)
            # GB-style banded branches (gb + the fixed-dimensional vgb) need
            # their band_info sub-state initialized; the real per-band
            # temperature ladders are set later in build_gb_moves /
            # build_vgb_moves.
            for _banded in ("gb", "vgb"):
                if _banded not in inds:
                    continue
                _nt_banded = self._branch_ntemps(_banded)
                self.logger.info(
                    "fresh start: band info '%s' (RSS %.0f MB)", _banded, _rss_mb()
                )
                band_temps = np.zeros(
                    (len(self.curr.source_info[_banded].band_edges) - 1, _nt_banded)
                )
                state.sub_states[_banded].initialize_band_information(
                    self.nwalkers,
                    _nt_banded,
                    self.curr.source_info[_banded].band_edges,
                    band_temps,
                )

            state.log_like = np.zeros((self.ntemps, self.nwalkers))
            state.log_prior = np.zeros((self.ntemps, self.nwalkers))
            # self.logger.debug("pickle state load success")

        # Sub-states that arrived without a tempered block (e.g. a resumed
        # file written before this branch had one) initialize from the main
        # state's ensemble.
        if state is not None and getattr(state, "sub_states", None):
            for _name, _sub in state.sub_states.items():
                if _sub is not None and not _sub.tempered_initialized:
                    _sub.pull_from_main(state, _name)

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
        pin_main_device(xp, general_info.gpus)

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

        # Walker -> owning device, mirroring AnalysisContainerArray's
        # contiguous ``np.array_split`` (analysiscontainer.py). Each walker's
        # data + sensitivity is then BUILT on the device that will own its
        # shard, so no later op touches an array resident on another device.
        # Without this the whole per-walker sensitivity (forward sens_mat,
        # detC) is allocated on the current device (gpus[0]) while the ACA
        # assigns half the walkers to gpus[1], and every subsequent read
        # (diagnostic.py noise term, domains.py residual add, the linalg.inv
        # batch) silently trips cupy's automatic peer access -- slow, and a
        # hard failure on nodes without P2P.
        _gpus_for_split = general_info.gpus
        _walker_device = {}
        if _gpus_for_split is not None and len(_gpus_for_split) > 1:
            for _s, _blk in enumerate(
                np.array_split(np.arange(self.nwalkers), len(_gpus_for_split))
            ):
                for _w in _blk:
                    _walker_device[int(_w)] = int(_gpus_for_split[_s])

        def _build_walker_ac(w):
            """Build walker ``w``'s AnalysisContainer (data + sensitivity).

            Called inside the walker's owning-device context so every array
            is allocated on the device that will hold its shard.
            """
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
                # NO transform_fn= here: psd_params (like galfor/sgwb above)
                # is ALREADY in the physical basis. The backend applies
                # transform_fn itself (sensitivity.py
                # SensitivityBackendBase.__call__), so passing both would
                # transform twice -- invisible while every stock psd
                # transform was None, wrong as soon as one is set (a
                # log-sampled branch would exponentiate exp(ln S)).
                sens_here = general_info.sensitivity_backend(
                    f"walker_{w}",
                    psd_params,
                    galfor_params=galfor_params,
                    **extra_sens_kwargs,
                )
            else:
                sens_here = general_info.sensitivity_backend(
                    f"walker_{w}", **general_info.fixed_psd_kwargs
                )

            return AnalysisContainer(
                deepcopy(data_res_arr),
                deepcopy(sens_here),
                signal_gen=dict(signal_gen_map) if signal_gen_map else None,
                likelihood_source_only=ll_source_only,
            )

        acs_tmp = []
        self.logger.info(
            "setup_acs: building %d walker ACs (RSS %.0f MB)",
            self.nwalkers, _rss_mb(),
        )
        for w in range(self.nwalkers):
            with device_context(xp, _walker_device.get(w)):
                acs_tmp.append(_build_walker_ac(w))
            if w % 8 == 7 or w == self.nwalkers - 1:
                self.logger.info(
                    "setup_acs: walker AC %d/%d built (RSS %.0f MB)",
                    w + 1, self.nwalkers, _rss_mb(),
                )

        gpus = general_info.gpus
        if gpus is not None and len(gpus) > 1 and self.nwalkers % len(gpus) != 0:
            logger.warning(
                "nwalkers=%d is not divisible by len(gpus)=%d: contiguous "
                "np.array_split shards are uneven, so per-shard batch sizes "
                "differ and any fixed-block intra-shard indexing is invalid "
                "(GBGPU uses rank-based indexing and stays correct). Prefer "
                "nwalkers %% ngpus == 0 for balanced device loads.",
                self.nwalkers, len(gpus),
            )
        acs = AnalysisContainerArray(
            acs_tmp,
            gpus=gpus,
            # Overlap per-split work (vectorized dispatch / signal_operation)
            # across devices; single-GPU/CPU runs stay serial.
            run_threaded=gpus is not None and len(gpus) > 1,
        )

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
                # Generate + subtract this walker's templates on the device
                # that owns its shard: ``ac.data`` is a view into the ACA's
                # shard buffer (on gpus[split]), so building the template and
                # the in-place ``add_signal`` (domains.py residual add) both
                # run on that device -- no cross-device peer access.
                with device_context(xp, _walker_device.get(w)):
                    gen_map = getattr(ac, "_signal_gen", None)
                    if not isinstance(gen_map, dict):
                        continue  # this walker's branches use the fallback below
                    params = {}
                    params_pre_transformed = []
                    for name in self.curr.engine_info.branch_names:
                        if name in ("psd", "galfor") or name not in gen_map:
                            continue
                        inds_w = state.branches_inds[name][0, w]
                        if not inds_w.any():
                            continue
                        rows = state.branches_coords[name][0, w][inds_w]
                        tf = getattr(self.curr.source_info.get(name), "transform", None)
                        if getattr(tf, "n_leaf_fills", None) is not None:
                            # PER-LEAF transform fills (e.g. EMRI xI0): the leaf
                            # identity of each row is needed, so pre-transform
                            # here and hand the generator waveform-basis rows.
                            leaf_ids = np.where(inds_w)[0]
                            params_pre_transformed.append(
                                (name, tf.both_transforms(rows, leaf_inds=leaf_ids))
                            )
                        else:
                            params[name] = rows
                    handled_by_signal_gen.update(params.keys())
                    handled_by_signal_gen.update(
                        name for name, _ in params_pre_transformed
                    )
                    if params:
                        template = ac.build_template(params)
                        # breakpoint()  # debug hook: inspect template vs ac.data here
                        ac.data.add_signal(template, sign=-1)
                    for name, phys_rows in params_pre_transformed:
                        template = ac.build_template(
                            {name: phys_rows}, apply_transform=False
                        )
                        ac.data.add_signal(template, sign=-1)

            # stft_tof fallback for branches without a registered generator.
            # TODO: add a vgb signal_gen rebuild hook — the per-leaf fill
            # container makes it trivial (coords + leaf_inds); until then
            # vgb follows the GB precedent (setup-time subtraction only).
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

            logger.info("rebuilt residuals from state coords/inds.")

        # One-time build->sampling reclamation (memory-lifecycle rule): the
        # residual/PSD plane now lives in the ACA's persistent shard buffers,
        # so drop the data processor's production transients and sweep every
        # device's memory pool ONCE. Never repeated during sampling; never
        # touches ACA/DCGA persistent allocations.
        proc = getattr(general_info, "data_processor", None)
        release = getattr(proc, "release_transients", None)
        if callable(release):
            release()
        if _xp_is_cupy:
            try:
                if gpus is not None:
                    for dev in gpus:
                        with xp.cuda.Device(int(dev)):
                            xp.get_default_memory_pool().free_all_blocks()
                else:
                    xp.get_default_memory_pool().free_all_blocks()
            except Exception as exc:  # cupy installed but no usable device
                logger.debug("post-production pool sweep skipped: %s", exc)

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
        self.logger.debug("state loaded (RSS %.0f MB)", _rss_mb())

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

        # Localize a non-finite initial likelihood before it trips Eryn's
        # opaque "initial log_like was +/- infinite". Reports, per shard,
        # whether the NON-finite values live in the residual buffers (a
        # waveform-production NaN) or the inverse-PSD buffers (a PSD /
        # sensitivity zero -> inf, e.g. the f=0 noise-model bin). Only runs
        # on the error path, so no cost to healthy runs.
        _ll0 = np.asarray(asnumpy(state.log_like[0]))
        if not np.all(np.isfinite(_ll0)):
            xp_a = acs.xp
            for si, (dbuf, pbuf) in enumerate(
                zip(acs.linear_data_arr, acs.linear_psd_arr)
            ):
                with (
                    xp_a.cuda.Device(int(acs.gpus[si]))
                    if acs.gpus is not None else _nullcontext()
                ):
                    d_bad = int(xp_a.sum(~xp_a.isfinite(dbuf)))
                    p_bad = int(xp_a.sum(~xp_a.isfinite(pbuf)))
                logger.warning(
                    "initial ll non-finite (shard %d): %d non-finite "
                    "residual value(s), %d non-finite invC value(s). "
                    "residual-side -> waveform-production NaN; invC-side "
                    "-> PSD/sensitivity zero (e.g. f=0 bin).",
                    si, d_bad, p_bad,
                )

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
        # Per-branch reset kwargs, routed to each sub-backend by name so two
        # GB-style branches (gb + vgb) do not clobber each other's
        # ``num_bands`` / ``band_edges`` in the flat merge below.
        sub_reset_kwargs = {}
        # Names the main reset call passes explicitly: keep them out of the
        # flat merge (sub-state reset_kwargs carry their own per-branch
        # ntemps/nwalkers/... geometry, which is routed via sub_reset_kwargs).
        _main_reset_names = {"ntemps", "nwalkers", "nleaves_max", "ndim", "ndims"}
        # TODO: fix this somehow
        for name in branch_names:
            if name in state.sub_states and state.sub_states[name] is not None:
                _rk = state.sub_states[name].reset_kwargs
                sub_reset_kwargs[name] = _rk
                extra_reset_kwargs = {
                    **extra_reset_kwargs,
                    **{
                        key: value
                        for key, value in _rk.items()
                        if key not in _main_reset_names
                    },
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
                sub_reset_kwargs=sub_reset_kwargs,
                **extra_reset_kwargs,
            )

            # Persist the domain settings (FD / STFT / WDM) so a re-run can
            # reconstruct everything from a single HDF5 file. ``general_info``
            # already holds the resolved instances at this point.
            domain_settings = general_info.domain_settings
            if domain_settings is not None:
                backend.write_domain_settings(domain_settings)

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

        # Recipe setup can change the residual (GB_SUBTRACT_OUT_OF_BAND /
        # subtract_neighbors remove known out-of-band catalogue GBs from acs),
        # and it runs AFTER the initial-logL print above. Re-evaluate so the
        # logged value -- and the sampler's starting state.log_like -- reflect
        # the post-subtraction residual (no-op when nothing subtracted).
        state.log_like[:] = acs.likelihood(complex=False)
        logger.info(f"initial log likelihood (after recipe setup): {state.log_like[0]}")

        # [layer-chi2 diag; GB_LAYER_CHI2=1] Where does the post-subtraction
        # residual live in frequency? Edge layers -> out-of-window source
        # leakage (not subtracted); center -> subtraction bug; even -> global.
        if os.environ.get("GB_LAYER_CHI2"):
            try:
                _ds = self.curr.general_info.domain_settings
                _res = np.asarray(asnumpy(acs.flatten()[0].data_res_arr.arr))
                if _res.ndim == 3:  # (nchannels, Nf, Nt)
                    _pl = (np.abs(_res) ** 2).sum(axis=(0, 2))  # -> (Nf,)
                    _ldf = float(getattr(_ds, "layer_df", 0.0))
                    _k0 = int(getattr(_ds, "_ind_min_f", 0))
                    _k1 = int(getattr(_ds, "_ind_max_f", len(_pl) - 1))
                    _lay = (list(range(_k0, _k1 + 1))
                            if len(_pl) == (_k1 - _k0 + 1) else list(range(len(_pl))))
                    _tot = float(_pl.sum()) or 1.0
                    logger.info("[layer-chi2] residual |r|^2 by WDM layer (total=%.4e, active %d..%d):",
                                _tot, _k0, _k1)
                    for _i, _L in enumerate(_lay):
                        if _pl[_i] > _tot * 1e-3:
                            logger.info("  layer %3d  f=%.6e Hz  |r|^2=%.4e  (%.1f%%)",
                                        _L, _L * _ldf, _pl[_i], 100.0 * _pl[_i] / _tot)
                else:
                    logger.warning("[layer-chi2] unexpected residual ndim=%d shape=%s",
                                   _res.ndim, _res.shape)
            except Exception as _e:  # diagnostic only, never break the run
                logger.warning("[layer-chi2] failed: %r", _e)

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

            self.sampler.run_mcmc(self.state, self.curr.general_info.num_iterations, thin_by=1, progress=self.progress, store=True)

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
