"""Top-level runner class (``GlobalFit``).

This module wires together the per-source samplers, the recipe driver, the
HDF backend, and MPI rank coordination into a runnable end-to-end global fit.

MPI rank roles (mirrors ``communication/ranks.py``'s docstring, which owns
the resolution: ``resolve_roles``/``build_layout``): HEAD (the main rank --
sequences the recipe, owns the full host ``GFState``, AND computes walker
block 0 like any other compute rank), COMPUTE (one device list and one
walker block each, running the ``ComputeService`` command loop), SAVER (the
highest non-head rank, at communicator size >= 3 only; below that the role
is aliased to the head and saves are synchronous). Every rank builds the
full pipeline; only the walker-block size of its ``AnalysisContainerArray``
differs. SPARE exists only under ``GF_LEGACY_RANK_LAYOUT=1``, which restores
the pre-port roles (one compute rank owning the whole per-node GPU pool,
every other non-saver rank built then stopped at startup without computing).

The legacy multi-stage pipeline classes (``GlobalFitSegment``,
``MPIControlGlobalFit``) were removed 2026-07 (parallel-resources plan P0).
"""

import hashlib
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

from eryn.moves import CombineMove
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
from ..coarsewdm import CoarseWDMRuntime, CoarseWDMStatistic, compute_qeff
from ..domains import CoarseWDMSettings, WDMSettings
from ..sensitivity import (
    CompositeSensitivityBackend,
    InstrumentNoise,
    UnequalArmInstrumentNoise,
)
from ..utils.device import device_context, pin_main_device
from ..utils.utility import asnumpy
from .communication.ranks import (
    RankRole,
    build_layout,
    install_mpi_abort_on_error,
    prefix_stdout,
    rank_tag,
    resolve_roles,
)
from .engine import (EngineInfo, GeneralSetup, GlobalFitEngine, GlobalFitInfo,
                     GlobalFitSettings, Setup)
from .hdfbackend import (GFHDFBackend, promote_backup_if_store_unreadable,
                         save_to_backend_asynchronously_and_plot)
from .loginfo import dump_settings, init_logger, setup_root_file_handler
from . import midit_checkpoint
from .moves import FunctionMove, GFCombineMove, GlobalFitMove, MoveBuildContext
from .postprocessing import GlobalFitPlotter, RunMetadata, SubmissionWriter, save_residuals
from .recipe import Recipe
from .state import GFState, make_cap_edges
from .utils import BasicResidualacsLikelihood


logger = getLogger(__name__)

#: Truthy/falsy spellings for :func:`null_check_only`, matching
#: ``stock.base.bool_or_str`` (imported nowhere here: ``stock`` builds ON
#: this module, so reaching back into it would be a cycle).
_NULL_CHECK_TRUE = ("1", "true", "yes", "on")


def null_check_only() -> bool:
    """``NULL_CHECK_ONLY=1``: measure the initial lnL, then stop.

    The truth-injection NULL TEST's entire output is ONE line --
    ``initial log likelihood (after recipe setup)`` -- so the per-source
    decomposition of it (18 jobs, one (branch, id) pair each) has nothing to
    sample. With this set, :meth:`GlobalFit.prepare_main` returns right after
    that print and :meth:`GlobalFit.run_global_fit` skips ``run_mcmc``.

    NOT a ``sys.exit``: the production layout is ``mpiexec -n 3`` (main +
    saver + spare), and the other two ranks sit in ``comm.recv``. The early
    return goes through the ORDINARY shutdown -- the spares' ``"stop"`` and
    the saver's ``{"finish_run": True}`` -- so every rank exits rc 0 in
    seconds instead of blocking until walltime.

    A present-but-empty value counts as unset (``stock.base._env_lookup``'s
    rule), so ``NULL_CHECK_ONLY= sbatch ...`` cannot silently skip a run.
    """
    return os.environ.get("NULL_CHECK_ONLY", "").strip().lower() in _NULL_CHECK_TRUE


def check_store_branch_ndims(stored_ndims, config_ndims, backend_path):
    """Refuse a resume whose stored per-branch ndim differs from the config.

    The CONSTRUCTION-level half of the resume-safety gate: ``stored_ndims``
    comes from the backend's ``ndims`` HDF5 *attributes* (a handful of
    scalars), so this fires before any chain is read. The identical check
    runs again on the loaded state further down :meth:`GlobalFit.load_info`
    as a backstop for stores whose attrs cannot be read.

    Only the INTERSECTION of the two branch sets is compared: adding or
    removing a branch between runs (``REMOVE_BRANCHES``) is legitimate and
    handled elsewhere.

    The vgb case gets its own message because it is the live one (user
    ruling 2026-09-16): ``VGB_CHIRP_MASS_BASIS=1`` moves Mc from the
    per-leaf fills to the sampled side, taking the vgb chain from 5 to 6
    columns. That is NOT resume-compatible — a store must be fresh, or
    migrated with ``scripts/fstat_proposal/migrate_vgb_chirp_basis.py``.
    Never flip the knob mid-store.

    Args:
        stored_ndims: ``{branch: ndim}`` as the store holds it (``None`` or
            empty is a no-op).
        config_ndims: ``{branch: ndim}`` the current run configuration builds.
        backend_path: Store path, quoted into the message.

    Raises:
        ValueError: A shared branch disagrees on ndim.
    """
    if not stored_ndims:
        return
    for name, cfg_nd in (config_ndims or {}).items():
        if name not in stored_ndims:
            continue
        stored_nd = int(stored_ndims[name])
        if stored_nd == int(cfg_nd):
            continue
        if name == "vgb":
            raise ValueError(
                f"Cannot resume {backend_path!r}: branch 'vgb' stored with "
                f"ndim {stored_nd} but the run config expects {int(cfg_nd)}. "
                "The VGB basis flag (VGB_CHIRP_MASS_BASIS: 5-dim legacy "
                "distance basis vs 6-dim chirp-mass basis, which samples Mc "
                "as well as fdot_astro_ratio) differs from the stored run. "
                "Either set VGB_CHIRP_MASS_BASIS to match the store, start a "
                "FRESH store, or migrate the file to the 6-dim basis with "
                "scripts/fstat_proposal/migrate_vgb_chirp_basis.py (never "
                "reshape silently, and never flip the knob mid-store)."
            )
        raise ValueError(
            f"Cannot resume {backend_path!r}: branch {name!r} stored with "
            f"ndim {stored_nd} but the run config expects {int(cfg_nd)}. For "
            "GB this usually means GB_USE_ASTROPHYSICAL_F0_MC_PRIOR / "
            "GB_USE_CHIRP_MASS differ from the stored run (8-col vs 9-col "
            "fdot_astro_ratio basis). Start a fresh backend or match the "
            "original config."
        )


def _leaf_moves(moves):
    """Flatten combine moves (recursively; ``(move, weight)`` tuples unwrapped).

    An EMPTY combine (no sub-moves) is itself returned as a leaf so a caller can
    still name it.
    """
    out = []
    for move in moves:
        if isinstance(move, tuple):
            move = move[0]
        inner = getattr(move, "moves", None) if isinstance(move, CombineMove) else None
        if inner:
            out.extend(_leaf_moves(inner))
        else:
            out.append(move)
    return out


def _move_label(move):
    return getattr(move, "gf_move_name", None) or getattr(move, "name", None) or type(move).__name__


def _fanout_unready_moves(moves):
    """``(unready, head_only)`` leaf-move names for a multi-rank run.

    ``unready``: ``GlobalFitMove`` leaves whose class still has the default
    ``gf_serve`` (they must be ported before running with several compute
    ranks). ``head_only``: every leaf that runs on the HEAD ALONE -- leaves
    that are not ``GlobalFitMove`` at all (plain eryn moves), ``FunctionMove``
    and anything that opts in via ``gf_head_only``. They are named in the
    warning rather than silently skipped: "head-only" is a real semantic
    change under multi-rank and the operator has to see which moves take it.

    "Head-only" means two DIFFERENT things, and the difference matters:

    * a ``FunctionMove``'s ``fn`` reads the head's ACA, which under the
      walker-block layout holds this rank's B rows only -- so it really does
      act on the head's walker block alone;
    * a plain eryn move gets the head's ``GFState``, which is the FULL
      ensemble (all N walkers). A zero-likelihood eryn move such as
      ``gb_ridge_gibbs`` / ``vgb_ridge_gibbs`` therefore acts on EVERY
      walker's coords and never touches an ACA at all.

    The one real caveat is the combination: a plain eryn move that DID call
    the likelihood would score all N walkers against the head's B-row ACA, so
    such a move must be ported to the fan-out before it is used here.

    The addremove (MBH/EMRI/SOBBH) and PSD (psd/galfor/sgwb) families are
    served through ``moves.walkerfanout.WalkerFanoutMixin`` since Plan 3;
    GB/VGB (GBSpecialBase) serve the three-command protocol since Plan 4;
    the FD dev-search setups raise under several ranks.
    """
    unready, head_only = [], []
    for move in _leaf_moves(moves):
        # set ``gf_head_only = True`` on a GlobalFitMove subclass to declare it
        # head-only; nothing sets it today.
        if isinstance(move, FunctionMove) or getattr(move, "gf_head_only", False):
            head_only.append(_move_label(move))
            continue
        if not isinstance(move, GlobalFitMove):
            head_only.append(_move_label(move))
            continue
        if type(move).gf_serve is GlobalFitMove.gf_serve:
            unready.append(_move_label(move))
    return unready, head_only


def _materialized_moves(recipe):
    """Every move across a materialized recipe's steps (flat, order-preserving).

    ``recipe.recipe`` is the runtime step list built by
    :meth:`~lisatools.globalfit.recipe.Recipe.add_recipe_component`: dicts of
    ``{"name", "adjust", "status"}`` whose ``adjust`` is a ``RecipeStep``
    holding ``moves``. Defensive on both counts -- a legacy settings-file
    recipe can register any ``setup_run``/``stopping_function`` object, and
    ``RecipeStep.moves`` RAISES when the step was never given moves.
    """
    out = []
    for step in getattr(recipe, "recipe", None) or []:
        adjust = step.get("adjust") if isinstance(step, dict) else step
        try:
            moves = list(getattr(adjust, "moves", None) or [])
        except Exception:  # noqa: BLE001 - RecipeStep.moves raises when unset
            continue
        out.extend(moves)
    return out


def _serve_registry(recipe):
    """Compute-rank serve registry ``{(step_name, move_name): leaf, name: leaf}``.

    Keyed from the materialized STEP LIST (``recipe.recipe``: dicts of
    ``{"name", "adjust", ...}``), never from ``gf_stage_name`` -- a stock
    runtime move is one object shared by every stage that lists it and the
    stamp keeps only the last stage. The head addresses a move by its
    ``gf_move_name`` with the current stage in the command clock
    (``ComputeService.handle`` resolves the pair first, then the bare name),
    so every (stage, name) pair a stage lists is registered; a bare-name entry
    is added only when the name maps to exactly ONE object across all stages
    (the safe fallback for a command whose clock carries no stage). Two
    DIFFERENT objects under one (stage, name) is a recipe defect and raises.
    """
    registry = {}
    objects_by_name = {}
    for step in getattr(recipe, "recipe", None) or []:
        step_name = step.get("name") if isinstance(step, dict) else None
        adjust = step.get("adjust") if isinstance(step, dict) else step
        try:
            moves = list(getattr(adjust, "moves", None) or [])
        except Exception:  # noqa: BLE001 - RecipeStep.moves raises when unset
            continue
        for leaf in _leaf_moves(moves):
            name = getattr(leaf, "gf_move_name", None)
            if name is None:
                continue
            key = (step_name, name)
            prior = registry.get(key)
            if prior is not None and prior is not leaf:
                raise RuntimeError(
                    f"two different moves named {name!r} in stage {step_name!r}: the "
                    "compute-rank serve registry cannot address both; give them distinct names."
                )
            registry[key] = leaf
            objects_by_name.setdefault(name, [])
            if all(obj is not leaf for obj in objects_by_name[name]):
                objects_by_name[name].append(leaf)
    for name, objs in objects_by_name.items():
        if len(objs) == 1:
            registry[name] = objs[0]
    return registry


def _rank_log_filenames(layout, rank):
    """(root lisatools log, GlobalFit log) for this rank; the head keeps today's names."""
    if int(rank) == int(layout.head_rank):
        return "globalfit_run.log", "global_fit.log"
    return f"globalfit_run.rank{int(rank)}.log", f"global_fit.rank{int(rank)}.log"


def _rebuild_state_view(state, walker_block):
    """The state the residual rebuild reads: the full state, or this rank's walker slice."""
    if walker_block is None:
        return state
    from .communication.walkerslice import slice_state

    w0, w1 = walker_block
    return slice_state(state, w0, w1, sub_states=[])


def _branch_cap_edges(branch_info):
    """Leaf-cap CELL edges for a banded branch (user design 2026-08-15).

    A refinement of the branch's ``band_edges`` by its ``cap_divisor``
    (``GBSettings.cap_divisor`` / ``GB_CAP_DIVISOR``). MUST agree with what
    the move is built with and with what the store holds: the state's
    resume guard and the move's arm-time check both compare against it.

    NOTE: do NOT assume "branches that are not GB lack the field" --
    ``VGBSettings`` SUBCLASSES ``GBSettings``, so it inherits
    ``cap_divisor`` and would silently pick up ``GB_CAP_DIVISOR``. VGB
    pins its own ``cap_divisor = 1`` (it is fixed-dimensional: no RJ, no
    leaf caps) precisely so this function and ``recipe.build_vgb_moves``
    -- which initializes on the plain band grid -- agree. Any future
    banded branch must make the same choice explicitly.
    """
    return make_cap_edges(
        branch_info.band_edges,
        int(getattr(branch_info, "cap_divisor", 1) or 1),
        stagger=bool(getattr(branch_info, "cap_stagger", False)),
    )

#: Branches with hand-written initialization in :meth:`GlobalFit.load_info`.
#: Anything else is seeded by the metadata-driven generic path there.
_LOAD_INFO_NAMED_BRANCHES = ("psd", "galfor", "sgwb", "mbh", "emri", "sobbh", "gb")


def seed_injection_coords(
    inj, factor, ntemps, nwalkers, additive_start_widths=None
):
    """Injection-seeded start coords for one branch, START_FACTOR convention.

    MULTIPLICATIVE scatter ``x * (1 + factor * randn)``: each parameter is
    perturbed by a FRACTION of its own value, so dimensions of wildly
    different magnitude (e.g. GB/VGB fdot ~1e-16 alongside a ln-amplitude
    ~-50) all scatter sensibly off the injection without a per-dimension
    covariance/width scale. ``factor = 0`` -> exact injection (truth-null
    checks).

    ``additive_start_widths`` (``{column index: width}``, branch settings
    metadata) marks columns whose truth is exactly 0: there the
    multiplicative form gives zero ensemble spread, and an affine-invariant
    stretch move can never create spread it does not have. Those columns get
    ``truth + factor * width * randn`` instead — the documented exception to
    the multiplicative ruling for exactly-zero truths (``factor = 0`` still
    reproduces the exact injection).

    Args:
        inj: ``(nleaves, ndim)`` injection rows.
        factor: The branch's ``<BRANCH>_START_FACTOR`` value.
        ntemps: Leading temperature-axis size of the returned block.
        nwalkers: Walker-axis size.
        additive_start_widths: Optional ``{column: additive width}``.

    Returns:
        ``(ntemps, nwalkers, nleaves, ndim)`` start coordinates.
    """
    inj = np.asarray(inj, dtype=float)
    nleaves, ndim = inj.shape
    coords = inj[None, None] * (
        1.0 + factor * np.random.randn(ntemps, nwalkers, nleaves, ndim)
    )
    for col, width in (additive_start_widths or {}).items():
        coords[..., int(col)] = inj[None, None, :, int(col)] + (
            factor
            * float(width)
            * np.random.randn(ntemps, nwalkers, nleaves)
        )
    return coords


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

    #: multi-rank runtime (set by ``communication.ranks.prepare_rank`` before the
    #: build and by ``GlobalFit`` at run time; ``None`` in single-process use)
    rank_layout = None
    fanout = None
    rank = None
    rank_device_mode = None

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
        """Legacy-compat wrapper over ``communication.ranks.resolve_roles``.

        Kept for old callers of the ``(main_rank, results_rank, spare_ranks)``
        shape: ``main_rank`` runs the sampler; at ``size >= 3`` the highest
        remaining rank becomes the dedicated results/saver rank (below that,
        saving is synchronous on main). ``resolve_roles`` never reads
        ``GF_LEGACY_RANK_LAYOUT`` (only ``communication.ranks.build_layout``
        does), so ``spare_ranks`` here is ALWAYS ``[]`` -- including under the
        legacy layout, where every non-head, non-saver rank IS in fact a
        stopped spare. A caller that needs the real spare set (or any other
        legacy-layout distinction) must go through
        ``communication.ranks.prepare_rank`` + ``layout.role_of(rank) ==
        RankRole.SPARE`` instead of this method.

        Exposed as a classmethod so launchers (``scripts/run_global.py``)
        can decide which ranks need the heavy data build *before*
        constructing :class:`GlobalFit`.

        Args:
            comm: MPI communicator.
            main_rank: Rank that drives the sampler. Default 0.

        Returns:
            ``(main_rank, results_rank, [])``.
        """
        head, saver, _compute = resolve_roles(comm.Get_size(), main_rank)
        return head, saver, []

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
        self.head_rank = self.curr.rank_info.head_rank
        # Layout: resolved before the build by prepare_rank (drivers /
        # StockGlobalFit.run). The late path covers fit.sample() and legacy
        # settings-file runs: single process, or CPU, or the legacy switch.
        layout = getattr(self.curr, "rank_layout", None)
        if layout is None:
            layout = build_layout(
                self.comm,
                self.nwalkers,
                list(self.curr.general_info.gpus or []),
                main_rank=int(self.curr.rank_info.main_rank),
            )
            if not layout.is_single() and self.curr.general_info.gpus:
                self.logger_early_warning = (
                    "multi-rank layout resolved AFTER the build: device pinning "
                    "did not run before the build allocated. Call "
                    "communication.ranks.prepare_rank(fit, comm) before fit.build()."
                )
            self.curr.rank_layout = layout
        self.layout = layout
        self.curr.rank = self.rank
        self.role = layout.role_of(self.rank)
        self.main_rank = layout.head_rank
        self.results_rank = layout.saver_rank
        self.compute_ranks = tuple(layout.compute_ranks)
        self.worker_ranks = tuple(layout.worker_ranks)
        self.ranks_to_give = [
            r for r in self.all_ranks if layout.role_of(r) == RankRole.SPARE
        ]
        self.fanout_comm = layout.make_fanout_comm(self.comm) if not layout.is_single() else None
        # THE symmetric site for the two sub-splits (2026-09-23). Every rank
        # reaches this line exactly once, which is what a collective needs;
        # ``WalkerFanout`` is NOT such a site (the head builds one while the
        # workers build a ``ComputeService`` instead), so the comms are built
        # here and passed in. ``group_comm`` carries work shared only by a
        # walker block's R replicas -- GB's per-unit delta ledger -- and
        # ``reps_comm`` carries anything per-WALKER rather than per-rank.
        # Both are ``None`` for a single compute rank, where neither exists
        # and the direct-call path must stay untouched.
        # Guarded on COMPUTE-RANK MEMBERSHIP, not on ``fanout_comm is not
        # None``: the saver's ``make_fanout_comm`` returns a NULL comm object
        # (not ``None``), and splitting a null comm raises. It is not a
        # member of the fan-out comm, so it takes neither sub-split.
        self.group_comm = None
        self.reps_comm = None
        if (self.fanout_comm is not None
                and int(self.rank) in tuple(layout.compute_ranks)
                and hasattr(layout, "make_group_comm")):
            self.group_comm = layout.make_group_comm(self.fanout_comm)
            self.reps_comm = layout.make_reps_comm(self.fanout_comm)
        if isinstance(self.comm, MPI.Comm) and self.comm.Get_size() > 1:
            install_mpi_abort_on_error(self.comm)

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
        root_log, gf_log = _rank_log_filenames(layout, self.rank)
        setup_root_file_handler(artifacts_dir, level=level, filename=root_log)
        self.logger = init_logger(
            filename=gf_log, level=level, name=name, log_dir=artifacts_dir,
            console=self.verbose,
        )
        if self.rank != self.main_rank:
            prefix_stdout(rank_tag(layout, self.rank))
        if getattr(self, "logger_early_warning", None):
            self.logger.warning(self.logger_early_warning)
        self.logger.info("%s\nthis rank: %s", layout.describe(), rank_tag(layout, self.rank))
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

    @staticmethod
    def _read_stored_branch_ntemps(backend, branch_names, logger_=None) -> dict:
        """``{branch: ntemps}`` from the store's sub-backend attrs.

        A cheap attrs read (no chain load) of the rung count each branch's
        sub-state was WRITTEN with. Missing branches / groups / attrs are
        simply absent from the result; any failure returns what was read so
        far (this only feeds the mid-iteration checkpoint gate, which then
        falls back to the configured count). Never raises.
        """
        out = {}
        try:
            with backend.open("r") as f:
                grp = f[backend.name]["sub_backend"]
                for name in branch_names:
                    if name in grp and "ntemps" in grp[name].attrs:
                        out[str(name)] = int(grp[name].attrs["ntemps"])
        except Exception as exc:  # noqa: BLE001 -- never block a resume
            if logger_ is not None:
                logger_.debug("stored branch ntemps attrs unreadable (%r)", exc)
        return out

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

    def _midit_checkpoint_validate(self, state) -> typing.Tuple[bool, str]:
        """Config-compatibility gate for a mid-iteration checkpoint state.

        A checkpoint is disposable: on ANY mismatch with the current run
        configuration (branch set, walker/leaf/dim shapes, per-branch
        temperature ladders, banded-branch band grids) it is rejected so
        the run falls back to the normal resume paths, instead of
        crashing later at the first save into differently-shaped
        datasets (the 2026-08-27 6-temp-era-store vs 4-temp-relaunch
        landmine, but for the pickle sidecar).
        """
        try:
            names = set(self.engine_info.branch_names)
            st_names = set(getattr(state, "branches", {}).keys())
            if names != st_names:
                return False, (
                    f"branch set {sorted(st_names)} != config {sorted(names)}"
                )
            ll = np.asarray(state.log_like)
            if ll.shape != (self.ntemps, self.nwalkers):
                return False, (
                    f"engine log_like shape {ll.shape} != "
                    f"{(self.ntemps, self.nwalkers)}"
                )
            sub_states = getattr(state, "sub_states", None) or {}
            for name in names:
                coords = state.branches[name].coords
                exp_tail = (
                    self.engine_info.nleaves_max[name],
                    self.curr.ndims[name],
                )
                if coords.shape[1] != self.nwalkers or coords.shape[-2:] != exp_tail:
                    return False, (
                        f"branch {name!r} coords {coords.shape} vs "
                        f"nwalkers={self.nwalkers}, (nleaves, ndim)={exp_tail}"
                    )
                sub = sub_states.get(name)
                if sub is None:
                    continue
                betas_all = getattr(sub, "betas_all", None)
                if betas_all is not None:
                    # The rung count the resume will ACTUALLY build at: the
                    # store's own (recipe.resume_ladder_wins: the stored
                    # ladder wins over {BRANCH}_NTEMPS on a resume), else the
                    # configuration. Comparing to the configuration alone
                    # rejected every checkpoint of a store born at 8 rungs
                    # once the script went back to 12 (2026-09-17: seven
                    # spot requeues in a row each restarted the iteration
                    # from the HDF store, zero progress in nine hours).
                    _stored_nt = (getattr(self, "_stored_branch_ntemps", None)
                                  or {}).get(name)
                    nt_branch = int(
                        _stored_nt if _stored_nt else self._branch_ntemps(name)
                    )
                    if np.asarray(betas_all).shape[-1] != nt_branch:
                        return False, (
                            f"branch {name!r} ladder has "
                            f"{np.asarray(betas_all).shape[-1]} rungs; the "
                            f"resume builds {nt_branch} "
                            f"({'store' if _stored_nt else 'config'})"
                        )
                band_info = getattr(sub, "band_info", None)
                if band_info and "band_edges" in band_info:
                    cfg_edges = getattr(
                        self.curr.source_info.get(name), "band_edges", None
                    )
                    if cfg_edges is not None:
                        stored = np.asarray(band_info["band_edges"])
                        cfg = np.asarray(cfg_edges)
                        if stored.shape != cfg.shape or not np.allclose(stored, cfg):
                            return False, (
                                f"branch {name!r} band grid differs from the "
                                f"run config (stored {stored.shape[0] - 1} "
                                f"bands vs config {cfg.shape[0] - 1})"
                            )
            return True, ""
        except Exception as exc:  # noqa: BLE001 -- reject, never crash resume
            return False, f"validation error: {exc!r}"

    @staticmethod
    def _noise_pin_window(branch: str):
        """The PHYSICAL support the run's own prior defines, per column.

        Read from the module that DEFINES the prior, never duplicated: a
        hand-rolled "sanity window" drifts, and the real constraint here is
        not plausibility but SUPPORT. A pin outside the branch's prior puts
        every walker and every rung at ``log_prior = -inf`` on iteration 0 --
        so a looser window that merely caught order-of-magnitude basis
        mistakes would wave through a point that kills the run.

        ``GALFOR_ALPHA_MAX`` and ``GALFOR_FREQ_PRIOR`` move the box exactly as
        ``galfor_prior_dict`` does (one resolver, ``galfor_prior_ranges``).
        Returns ``()`` for an unknown branch (no check).
        """
        from .stock.erebor.noise import PSD_PRIOR_RANGE, galfor_prior_ranges

        if branch == "psd":
            return tuple(tuple(map(float, r)) for r in PSD_PRIOR_RANGE)
        if branch != "galfor":
            return ()
        return galfor_prior_ranges()

    def _seed_noise_start_coords(self, branch: str, raw: str, drawn):
        """``{PSD,GALFOR}_START_PARAMS`` -> start coords for a noise branch.

        ``raw`` is a comma list of PHYSICAL (linear) values; the return is in
        the branch's SAMPLING basis, broadcast over the drawn block's shape.
        Every walker and rung starts at the same point: these are 2- and
        5-parameter branches whose moves are ensemble proposals over a
        posterior that is already tight by the time they matter, and a pinned
        start is meant to BE a pin -- ``*_START_FACTOR`` scatter would defeat
        the purpose of pinning the maxlogL point of a converged run.

        Raises on a wrong length or an out-of-window value rather than
        starting the fit somewhere absurd: this knob's whole failure mode is
        a basis mistake, which is silent and costs a whole allocation.
        """
        want = np.asarray(drawn).shape[-1]
        try:
            vals = np.array([float(x) for x in raw.split(",") if x.strip()],
                            dtype=float)
        except ValueError as exc:
            raise ValueError(
                f"{branch.upper()}_START_PARAMS must be a comma list of "
                f"floats, got {raw!r}."
            ) from exc
        if vals.shape[0] != want:
            raise ValueError(
                f"{branch.upper()}_START_PARAMS has {vals.shape[0]} values "
                f"but the {branch!r} branch samples {want} parameters."
            )
        for i, (lo, hi) in enumerate(self._noise_pin_window(branch)):
            if not (lo <= vals[i] <= hi):
                raise ValueError(
                    f"{branch.upper()}_START_PARAMS[{i}] = {vals[i]:.6g} is "
                    f"outside the {branch} branch's PRIOR support "
                    f"[{lo:.3g}, {hi:.3g}], so every walker and every rung "
                    f"would start at log_prior = -inf. Two likely causes: "
                    f"(a) these knobs take LINEAR values and a raw chain row "
                    f"from a log-sampled run was pasted in (convert first: "
                    f"exp for psd, 10** for galfor's amp/fk/f_1/f_2; alpha "
                    f"is linear in both bases); (b) the source run predates "
                    f"2026-08-13 (ea7790f3), which pulled the galfor "
                    f"fk/f_1/f_2 ceilings in from the old slope-unit values "
                    f"-- such a point is not reachable inside today's box. "
                    f"``python -m lisatools.globalfit.warmstart.noise_pin`` "
                    f"performs both conversions and this check for you."
                )
        # Into the SAMPLING basis, mirroring prepare_psd_branch /
        # prepare_galfor_branch: psd is ln of both columns, galfor is log10 of
        # GALFOR_LOG_PARAMS only (alpha is an O(1) power-law index and stays
        # linear). The galfor column set is read from the module that defines
        # it rather than hardcoded, so a basis change cannot desynchronize
        # this from the prior and the transform.
        info = self.curr.source_info.get(branch)
        log_sampling = bool(getattr(info, "log_sampling", False))
        sampled = vals.copy()
        if log_sampling:
            if branch == "psd":
                sampled = np.log(vals)
            else:
                from .stock.erebor.noise import (
                    GALFOR_BASIS, GALFOR_LOG_PARAMS)

                _cols = [GALFOR_BASIS.index(n) for n in GALFOR_LOG_PARAMS]
                sampled[_cols] = np.log10(vals[_cols])
        out = np.broadcast_to(
            sampled, np.asarray(drawn).shape).astype(float).copy()
        self.logger.info(
            "[NOISE-PIN] %s start coords PINNED at the supplied physical "
            "point %s -> sampling basis %s (log_sampling=%s); every walker "
            "and rung starts there, no START_FACTOR scatter.",
            branch, np.array2string(vals, precision=6),
            np.array2string(sampled, precision=6), log_sampling,
        )
        return out

    def _source_warm_start(self, branch: str, coords, prior):
        """SOURCE WARM START for one fixed-leaf source branch (fresh start only).

        User request 2026-10-08: a leaf whose catalogue id is in the seed
        store (``SOURCE_WARM_START_STORE``) with seed-run optimal SNR above
        ``SOURCE_WARM_START_SNR_MIN`` starts at that store's last cold-chain
        positions; every other leaf keeps the ``*_START_FACTOR`` start already
        in ``coords``. Store unset -> ``coords`` is returned AS IS (same object,
        no file touched). Logs one ``[SOURCE-WARM]`` line per armed branch.
        See :mod:`lisatools.globalfit.warmstart.sources`.
        """
        info = self.curr.source_info.get(branch)
        store = getattr(info, "source_warm_start_store", None)
        if not store:
            return coords
        from .warmstart.sources import warm_start_branch

        out, line = warm_start_branch(
            branch,
            coords,
            store=str(store),
            seed_ids=getattr(info, "source_warm_start_ids", None),
            new_ids=getattr(info, "injection_ids", None),
            snr_min=float(getattr(info, "source_warm_start_snr_min", 10.0)),
            snr_unknown=str(getattr(info, "source_warm_start_snr_unknown", "refuse")),
            ntemps_branch=self._branch_ntemps(branch),
            prior=prior,
            data_t0=getattr(self.curr.general_info, "data_t0", None),
        )
        self.logger.info(line)
        return out

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
        # SELF-HEAL A TORN STORE BEFORE TOUCHING IT (2026-08-16). On the spot
        # partition a preemption can kill a job mid-write, leaving gzip
        # chunks that open fine and only fail when read -- so the next job
        # died at resume with "filter returned failure during read". This
        # promotes the running backup (validated first, damaged file kept as
        # *_CORRUPT_<n>.h5) so the run continues instead of needing a manual
        # swap. No-op when the store is healthy.
        if os.path.exists(backend_path):
            try:
                promote_backup_if_store_unreadable(backend_path)
            except Exception as _heal_err:      # never block a healthy start
                logger.warning("store self-heal check failed: %r", _heal_err)
        backend = None
        _stored_it = 0
        if os.path.exists(backend_path):
            backend = GFHDFBackend(
                backend_path,
                sub_state_bases=self.engine_info.branch_states,
                sub_backend=self.engine_info.branch_backends,
            )
            if getattr(backend, "initialized", False):
                _stored_it = int(getattr(backend, "iteration", 0) or 0)
            # CHEAPEST SEAM where branch ndim meets the store: the ``ndims``
            # HDF5 attrs, read at backend construction, before any chain
            # load and before the mid-iteration checkpoint path. Catches the
            # VGB_CHIRP_MASS_BASIS 5 <-> 6 flip against an existing store
            # immediately instead of minutes into a build. An unreadable
            # attrs group is NOT fatal here: the post-load backstop below
            # repeats the comparison on the loaded state.
            if _stored_it > 0:
                try:
                    _stored_ndims = dict(backend.ndims)
                except Exception as _nd_err:    # never block a healthy start
                    logger.debug(
                        "store ndims attrs unreadable (%r); the post-load "
                        "ndim guard remains", _nd_err)
                else:
                    check_store_branch_ndims(
                        _stored_ndims, self.curr.ndims, backend_path
                    )

        # Mid-iteration checkpoint (preemption protection): the sidecar
        # snapshot beats the HDF store when it was written at (or after) the
        # store's newest stored iteration -- it carries partial progress of
        # the iteration the store never got to save. Config-incompatible or
        # unreadable checkpoints are moved aside and the normal resume paths
        # below take over (fail safe, never crash).
        # The gate compares per-branch ladders against what the resume will
        # BUILD, which for a resumed store is the store's own rung count
        # (recipe.resume_ladder_wins), not the configured knob.
        self._stored_branch_ntemps = (
            self._read_stored_branch_ntemps(
                backend, self.engine_info.branch_names, logger_=self.logger
            )
            if backend is not None and _stored_it > 0 else {}
        )
        _ckpt = midit_checkpoint.load_for_resume(
            backend_path,
            _stored_it,
            validate=self._midit_checkpoint_validate,
            logger_=self.logger,
        )
        if _ckpt is not None:
            state, _ckpt_meta = _ckpt
            self.logger.info(
                "RESUMING from mid-iteration checkpoint %s (boundary %r, "
                "written with the store at iteration %d; store now holds %d): "
                "partial-iteration progress recovered; a fresh iteration "
                "starts from this state.",
                midit_checkpoint.checkpoint_path(backend_path),
                _ckpt_meta.get("tag"),
                int(_ckpt_meta.get("stored_iteration", -1)),
                _stored_it,
            )

        if backend is not None and state is None:
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
                # rows this job appends start here (the residual snapshot
                # labels itself with the store row, not the job's count)
                self._resume_store_iteration = int(backend.iteration)
                # Guard against resuming a backend whose per-branch sampled
                # dimensionality no longer matches the run config -- the most
                # likely cause is toggling GB_USE_ASTROPHYSICAL_F0_MC_PRIOR /
                # GB_USE_CHIRP_MASS (8 <-> 9 column GB basis), or
                # VGB_CHIRP_MASS_BASIS (5 <-> 6 column VGB basis), between
                # runs. BACKSTOP: the same comparison already ran against the
                # store's ``ndims`` attrs at backend construction above; this
                # repeat reads the dimensionality off the LOADED coords, so
                # it also covers a store whose attrs disagree with its data.
                check_store_branch_ndims(
                    {
                        _name: _coords.shape[-1]
                        for _name, _coords in (
                            getattr(state, "branches_coords", {}) or {}
                        ).items()
                        if _coords is not None
                    },
                    self.curr.ndims,
                    backend_path,
                )

                # Guard against resuming a backend whose banded-branch band
                # grid no longer matches the run config (e.g.
                # GB_BAND_EDGES_MODE / GB_BAND_TARGET_COUNT /
                # GB_SUBBAND_DIVISOR changed between runs). The stored
                # band_* arrays (band_temps, band_leaf_cap, swap/proposal
                # counters, band_num_binaries) are sized (num_bands, ...) —
                # without this gate the mismatch surfaces as a bare
                # AssertionError in GBState.initialize_band_information or a
                # silent mis-addressed flat cell index in the GB move.
                _sub_states = getattr(state, "sub_states", {}) or {}
                for _banded in ("gb", "vgb"):
                    _sub = _sub_states.get(_banded)
                    _bi = getattr(_sub, "band_info", None) if _sub is not None else None
                    if not _bi or "band_edges" not in _bi:
                        continue
                    try:
                        _cfg_edges = np.asarray(
                            self.curr.source_info[_banded].band_edges
                        )
                    except (KeyError, AttributeError, TypeError):
                        continue
                    _stored_edges = np.asarray(_bi["band_edges"])
                    if _stored_edges.shape != _cfg_edges.shape or not np.allclose(
                        _stored_edges, _cfg_edges
                    ):
                        # Drop the CONFIG-built edges next to the store so the
                        # migration consumes exactly what this guard compared
                        # against (2026-08-22: deriving them offline risks a
                        # second refusal on any rounding difference — the
                        # VGB_BAND_LAYERS coarsening showed exactly that).
                        _edges_out = os.path.join(
                            os.path.dirname(os.path.abspath(backend_path)),
                            f"{_banded}_config_band_edges.npy",
                        )
                        try:
                            np.save(_edges_out, _cfg_edges)
                        except OSError:
                            _edges_out = "<could not write config edges>"
                        raise ValueError(
                            f"Cannot resume {backend_path!r}: branch "
                            f"{_banded!r} stored with "
                            f"{_stored_edges.shape[0] - 1} sub-bands but the "
                            f"run config builds {_cfg_edges.shape[0] - 1} "
                            f"(band edges differ). The band-edge knobs "
                            f"(GB_BAND_EDGES_MODE / GB_BAND_TARGET_COUNT / "
                            f"GB_BAND_MIN_LAYERS / GB_SUBBAND_DIVISOR"
                            f"{' / VGB_BAND_LAYERS' if _banded == 'vgb' else ''}) "
                            f"differ from the stored run. Either restore the "
                            f"original knobs, or migrate the file's per-band "
                            f"arrays onto the new band grid with scripts/"
                            f"fstat_proposal/migrate_gb_band_edges.py "
                            f"--branch {_banded} --edges-npy {_edges_out} "
                            f"(the config-built edges were just written "
                            f"there; band temperatures are interpolated; "
                            f"leaf caps and swap counters reset and "
                            f"re-earn; never reshape silently). Any in-move "
                            f"F-stat fit epoch cache is keyed by band index "
                            f"and must be refit — the loader refuses stale "
                            f"grids."
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
                cap_edges=_branch_cap_edges(self.curr.source_info["gb"]),
                leaf_cap_per_walker=bool(getattr(
                    self.curr.source_info["gb"], "leaf_cap_per_walker", False
                )),
                search_stage_per_walker=bool(getattr(
                    self.curr.source_info["gb"],
                    "search_stage_per_walker", False,
                )),
                search_shutoff_per_walker=bool(getattr(
                    self.curr.source_info["gb"],
                    "search_shutoff_per_walker", False,
                )),
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
            # NOISE START PIN (user ruling 2026-09-24: "the warmstart and
            # PSD/GB frozen start point come from the same folder ... it
            # should use maxlogL"). psd and galfor are the ONLY sampled
            # branches with no start-coordinate path at all -- they are in
            # ``_LOAD_INFO_NAMED_BRANCHES``, so the generic injection seeder
            # below skips them, and the arms above set ``inds`` only. Their
            # chains therefore begin at a PRIOR DRAW, which for a v9 search
            # whose first two stages do not sample noise would mean searching
            # against a random noise level for the whole of both stages.
            #
            # ⚠ BASIS. These knobs are PHYSICAL/LINEAR -- psd
            # ``[Soms_d, Sa_a]`` as square-root levels, galfor the 5-vector
            # ``(amp, fk, alpha, f_1, f_2)`` -- and are converted here into
            # whatever basis the branch SAMPLES in, exactly as
            # ``prepare_psd_branch`` converts the injection (psd is ``ln``,
            # galfor is ``log10``, and galfor's ``alpha`` column stays
            # linear). Taking physical values is what makes the knob safe to
            # fill from another run whose log-sampling flags differ from this
            # one's; a raw chain row pasted straight in would be silently
            # double-logged.
            for _nb in ("psd", "galfor"):
                if _nb not in coords:
                    continue
                _raw = os.environ.get(f"{_nb.upper()}_START_PARAMS", "").strip()
                if not _raw:
                    continue
                coords[_nb] = self._seed_noise_start_coords(
                    _nb, _raw, coords[_nb]
                )
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
                coords[key] = seed_injection_coords(
                    inj,
                    factor,
                    nt_draw,
                    self.nwalkers,
                    additive_start_widths=getattr(
                        self.curr.source_info.get(key),
                        "additive_start_widths",
                        None,
                    ),
                )

            # SOURCE WARM START (user 2026-10-08): leaves over the seed SNR
            # threshold start at the seed store's last cold-chain ladder
            # instead of the injection; off (store unset) returns the block
            # untouched. Must run before the per-branch ladder slice below.
            for _sb in ("mbh", "emri", "sobbh"):
                if _sb in coords:
                    coords[_sb] = self._source_warm_start(_sb, coords[_sb], priors.get(_sb))

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
                    cap_edges=_branch_cap_edges(
                        self.curr.source_info[_banded]
                    ),
                    # VGB has no such setting (and no RJ surface, so no
                    # caps at all) -- getattr keeps it False there.
                    leaf_cap_per_walker=bool(getattr(
                        self.curr.source_info[_banded],
                        "leaf_cap_per_walker", False,
                    )),
                    # Same getattr rule: VGB has neither setting.
                    search_stage_per_walker=bool(getattr(
                        self.curr.source_info[_banded],
                        "search_stage_per_walker", False,
                    )),
                    search_shutoff_per_walker=bool(getattr(
                        self.curr.source_info[_banded],
                        "search_shutoff_per_walker", False,
                    )),
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

    def _prepare_coarse_wdm_runtime(self, state: GFState):
        """Finalize the CPU-only coarse WDM likelihood once state exists."""
        general_info = self.curr.general_info
        Q = int(getattr(general_info, "coarse_Q", 1) or 1)
        if Q <= 1:
            return None

        existing = getattr(general_info, "coarse_wdm_statistic", None)
        if existing is not None:
            return existing

        allowed = {"psd", "galfor", "sgwb"}
        branches = set(self.curr.engine_info.branch_names)
        if "psd" not in branches:
            # A run with no psd branch samples no noise parameters, and the
            # coarse WDM machinery exists ONLY to accelerate the PSD noise
            # likelihood -- skip it and keep the fine backend, instead of
            # refusing to run. Hit in production 2026-09-16: the nogb NULL
            # test (psd removed, fixed noise params, source-only lnL)
            # inherits the main campaign's COARSE_* knobs and died here at
            # launch.
            logger.warning(
                "coarse_Q=%d requested but this run has no 'psd' branch "
                "(branches=%s): the coarse WDM noise likelihood only "
                "accelerates PSD sampling, so it is skipped and the fine "
                "backend stands.", Q, sorted(branches))
            return None
        unsupported = sorted(branches - allowed)
        mode = str(getattr(general_info, "coarse_gpu_mode", "off") or "off")
        all_source_sidecar = bool(unsupported)
        if all_source_sidecar and mode == "off":
            raise ValueError(
                "Coarse WDM noise likelihood cannot replace the fine backend "
                f"with source branches {unsupported} present; its statistic "
                "would go stale against per-walker residuals. An all-source "
                "run must opt into the sidecar runtime explicitly: set "
                "COARSE_GPU_MODE='search_approx' (optimization stages) or "
                "'delayed_acceptance' (production PE)."
            )
        if not all_source_sidecar and mode != "off":
            raise ValueError(
                "coarse_gpu_mode applies to all-source runs only; noise-only "
                "runs use the CPU backend-replacement coarse path."
            )
        # (psd-less compositions returned above -- psd is guaranteed here)
        if not all_source_sidecar and general_info.gpus is not None:
            raise ValueError(
                "Coarse WDM noise likelihood is CPU-only in this implementation; "
                "unset general.gpus. Single-GPU support is a planned follow-up."
            )
        fine_settings = general_info.domain_settings
        if not isinstance(fine_settings, WDMSettings) or isinstance(
            fine_settings, CoarseWDMSettings
        ):
            raise TypeError("coarse_Q > 1 requires a fine WDMSettings data domain.")
        if getattr(fine_settings, "is_complex", False):
            raise ValueError("Coarse WDM likelihood currently supports real WDM only.")

        fine_backend = general_info.sensitivity_backend
        if not isinstance(fine_backend, CompositeSensitivityBackend):
            raise TypeError(
                "Coarse WDM noise likelihood currently requires "
                "CompositeSensitivityBackend."
            )
        backend_name = str(getattr(fine_backend.backend, "name", ""))
        if not all_source_sidecar and not backend_name.endswith("_cpu"):
            raise ValueError(
                f"Coarse WDM noise likelihood is CPU-only; got backend {backend_name!r}."
            )

        policy = str(getattr(general_info, "coarse_fiducial", "injection"))
        if policy not in ("injection", "initial"):
            raise ValueError(
                f"coarse_fiducial must be 'injection' or 'initial'; got {policy!r}."
            )
        use_ws = bool(getattr(general_info, "coarse_use_ws", True))

        def _initial_median(name):
            rows = np.asarray(state.branches_coords[name][0, :, 0], dtype=float)
            transform = getattr(self.curr.source_info[name], "transform", None)
            if transform is not None:
                rows = np.asarray(transform.both_transforms(rows), dtype=float)
            return np.median(rows, axis=0)

        def _backend_has_samples():
            path = general_info.main_file_path
            if not os.path.exists(path):
                return False
            import h5py

            with h5py.File(path, "r") as handle:
                group = handle.get("global_fit") or handle.get("mcmc")
                return group is not None and int(group.attrs.get("iteration", 0)) > 0

        def _initial_fiducial_params():
            """Load or atomically freeze the first-run physical medians."""
            path = os.path.join(
                general_info.artifacts_file_dir, "coarse_wdm_initial_fiducial.npz"
            )
            expected_has_galfor = "galfor" in branches
            expected_has_sgwb = "sgwb" in branches
            if os.path.exists(path):
                with np.load(path, allow_pickle=False) as saved:
                    saved_q = int(saved["coarse_Q"][0])
                    saved_has_galfor = bool(saved["has_galfor"][0])
                    saved_has_sgwb = bool(saved["has_sgwb"][0])
                    if (
                        saved_q != Q
                        or saved_has_galfor != expected_has_galfor
                        or saved_has_sgwb != expected_has_sgwb
                    ):
                        raise ValueError(
                            f"Stored coarse fiducial {path!r} does not match this "
                            "run's Q/branches. Use a new tag or remove both the "
                            "backend and its artifact directory for a fresh run."
                        )
                    return (
                        np.asarray(saved["psd"], dtype=float),
                        np.asarray(saved["galfor"], dtype=float)
                        if expected_has_galfor
                        else None,
                        np.asarray(saved["sgwb"], dtype=float)
                        if expected_has_sgwb
                        else None,
                    )

            if _backend_has_samples():
                raise ValueError(
                    "Cannot resume a coarse_fiducial='initial' chain because "
                    f"its frozen parameter sidecar {path!r} is missing. "
                    "Recomputing it from the last walkers would change the target "
                    "likelihood; restore the sidecar or start under a new tag."
                )

            psd = _initial_median("psd")
            galfor = _initial_median("galfor") if expected_has_galfor else None
            sgwb = _initial_median("sgwb") if expected_has_sgwb else None
            os.makedirs(general_info.artifacts_file_dir, exist_ok=True)
            tmp_path = path + ".tmp.npz"
            np.savez(
                tmp_path,
                coarse_Q=np.asarray([Q], dtype=int),
                has_galfor=np.asarray([expected_has_galfor], dtype=bool),
                has_sgwb=np.asarray([expected_has_sgwb], dtype=bool),
                psd=psd,
                galfor=np.asarray([] if galfor is None else galfor, dtype=float),
                sgwb=np.asarray([] if sgwb is None else sgwb, dtype=float),
            )
            os.replace(tmp_path, path)
            return psd, galfor, sgwb

        if use_ws:
            if policy == "initial":
                psd_params, galfor_params, sgwb_params = _initial_fiducial_params()
            else:
                psd_params = np.asarray(general_info.psd_injection, dtype=float)
                # Same fallback as the sampled path: with the branch removed
                # but a foreground PINNED, the coarse surrogate's fiducial
                # should be that pinned shape, not a galaxy-free one. Delayed
                # acceptance stays exact either way, so a mismatch here costs
                # stage-1 screening efficiency rather than correctness.
                _gal_fixed = (general_info.fixed_psd_kwargs or {}).get("galfor_params")
                galfor_params = (
                    np.asarray(general_info.galfor_injection, dtype=float)
                    if "galfor" in branches
                    else (None if _gal_fixed is None else np.asarray(_gal_fixed, dtype=float))
                )
                sgwb_params = (
                    np.asarray(general_info.sgwb_injection, dtype=float)
                    if "sgwb" in branches
                    else None
                )

        else:
            # Bartlett weights are exactly the cell sizes and have no
            # fiducial dependence. Avoid both a fine covariance build and an
            # unnecessary initial-policy resume sidecar on this opt-out path.
            fiducial_fine = None
        coarse_settings = CoarseWDMSettings.from_fine(fine_settings, Q)

        backend_kwargs = dict(general_info.sensitivity_init_kwargs or {})
        coarse_backend = type(fine_backend)(
            settings=coarse_settings,
            force_backend=general_info.force_backend,
            **backend_kwargs,
        )
        # Reuse any fine unit-noise bases already paid for by the fiducial
        # build. Unequal-arm coarse bases add their averaged entries here.
        if getattr(fine_backend, "_instrument_basis_cache", None) is not None:
            coarse_backend._instrument_basis_cache = fine_backend._instrument_basis_cache

        qeff = qeff_channels = None
        fused_unequal = (
            use_ws
            and isinstance(fine_backend.instrument_component_cls, type)
            and issubclass(
                fine_backend.instrument_component_cls, UnequalArmInstrumentNoise
            )
        )
        if fused_unequal:
            # Build only the non-instrument part on the fine grid.  The exact
            # unequal-arm component streams its two unit bases directly into
            # coarse cells and retains just their three fine diagonals, which
            # are enough to moment-match total instrument+foreground/SGWB
            # variances without ever materializing two dense fine 3x3 bases.
            extra_diagonal = None
            if galfor_params is not None or sgwb_params is not None:
                extra_fine = fine_backend(
                    "coarse_fiducial_noninstrument",
                    None,
                    galfor_params=galfor_params,
                    sgwb_params=sgwb_params,
                )
                extra_covariance = np.asarray(asnumpy(extra_fine.sens_mat))
                extra_diagonal = np.real(
                    np.stack([extra_covariance[a, a] for a in range(3)], axis=0)
                )
                del extra_covariance, extra_fine

            model = coarse_backend.instrument_model_cls(
                float(psd_params[0]) ** 2,
                float(psd_params[1]) ** 2,
                coarse_backend._orbits,
                f"{coarse_backend.model_name}:coarse_fiducial",
            )
            component_kwargs = dict(
                tdi_generation=coarse_backend.tdi_generation,
                model=model,
                fill_nans=coarse_backend.instrument_fill_nans,
                **coarse_backend.instrument_component_kwargs,
            )
            if coarse_backend._instrument_basis_cache is not None and issubclass(
                coarse_backend.instrument_component_cls, InstrumentNoise
            ):
                component_kwargs["basis_cache"] = coarse_backend._instrument_basis_cache
            component = coarse_backend.instrument_component_cls(**component_kwargs)
            qeff, qeff_channels = component.coarse_qeff(
                coarse_settings, extra_diagonal=extra_diagonal
            )
            fiducial_fine = None
            del extra_diagonal
        elif use_ws:
            fiducial_fine = fine_backend(
                "coarse_fiducial",
                psd_params,
                galfor_params=galfor_params,
                sgwb_params=sgwb_params,
            )

        if all_source_sidecar:
            # Fine backend stays canonical: build the runtime sidecar and
            # leave every AnalysisContainer's coarse_stats as None. The
            # per-walker statistics are refreshed by the noise moves.
            if qeff is None and use_ws:
                qeff, qeff_channels = compute_qeff(
                    fiducial_fine,
                    coarse_settings,
                    use_ws=True,
                    return_channels=True,
                )
            digest_src = hashlib.sha256()
            for part in (policy, str(Q), str(bool(use_ws)), mode):
                digest_src.update(part.encode())
            if use_ws:
                for arr in (psd_params, galfor_params, sgwb_params):
                    if arr is not None:
                        digest_src.update(
                            np.ascontiguousarray(
                                np.asarray(arr, dtype=float)
                            ).tobytes()
                        )
            runtime = CoarseWDMRuntime(
                coarse_settings=coarse_settings,
                qeff=qeff,
                qeff_channels=qeff_channels,
                use_ws=use_ws,
                mode=mode,
                batch_bytes=int(
                    getattr(general_info, "coarse_gpu_batch_bytes", 0)
                    or 256 * 1024 * 1024
                ),
                fiducial_digest=digest_src.hexdigest()[:16],
                coarse_backend=coarse_backend,
            )
            general_info.coarse_wdm_runtime = runtime
            general_info.coarse_wdm_settings = coarse_settings
            logger.info(
                "coarse WDM sidecar runtime (all-source, mode=%s): Q=%d, "
                "Nt_active=%d -> Ncoarse=%d, weighting=%s, fiducial=%s, "
                "digest=%s",
                mode,
                Q,
                fine_settings.Nt_active,
                coarse_settings.Ncoarse,
                "WS" if use_ws else "Bartlett",
                policy if use_ws else "not used",
                runtime.fiducial_digest,
            )
            return None

        statistic = CoarseWDMStatistic.from_wdm_signal(
            general_info.input_data_residual_array,
            coarse_settings,
            fiducial_sens_mat_fine=fiducial_fine,
            use_ws=use_ws,
            qeff=qeff,
            qeff_channels=qeff_channels,
        )

        general_info.fine_sensitivity_backend = fine_backend
        general_info.sensitivity_backend = coarse_backend
        general_info.coarse_wdm_settings = coarse_settings
        general_info.coarse_wdm_fiducial_sens_mat = fiducial_fine
        general_info.coarse_wdm_statistic = statistic
        logger.info(
            "coarse WDM likelihood: CPU Q=%d, Nt_active=%d -> Ncoarse=%d, "
            "weighting=%s, fiducial=%s",
            Q,
            fine_settings.Nt_active,
            coarse_settings.Ncoarse,
            "WS" if use_ws else "Bartlett",
            policy if use_ws else "not used",
        )
        return statistic

    def setup_acs(
        self,
        state: GFState,
        rebuild_residuals: bool = False,
        walker_block: typing.Optional[typing.Tuple[int, int]] = None,
    ) -> AnalysisContainerArray:
        """
        Set up AnalysisContainerArray for likelihood computations.

        Creates analysis containers for each walker, initializing data
        residuals and sensitivity curves. Domain dispatch (FD / STFT / WDM)
        flows through ``general_info.input_data_residual_array`` and the
        configured :class:`XYZSensitivityBackend`, so nothing in this
        method is FD-specific.

        Args:
            state: GFState object containing current parameter values. ALWAYS
                the full ``nwalkers``-walker state, on every rank -- the walker
                block narrows what is BUILT, never what is passed in. The only
                place the ``[w0, w1)`` slice is taken is the residual rebuild
                below (``_rebuild_state_view``), which needs one template row
                per local container.
            rebuild_residuals: If ``True``, subtract each non-PSD branch's
                current templates from the freshly-built containers so the
                stored arrays are residuals rather than raw data (stft_tof
                restart/handover path; it was disabled there while the EMRI
                branch was being debugged, so it stays opt-in here).
            walker_block: ``(w0, w1)`` global walker range this rank owns;
                ``None`` builds every walker (single-process behaviour).

        Returns:
            AnalysisContainerArray with one container per walker in
            ``[w0, w1)`` (every walker when ``walker_block`` is ``None``).
        """
        general_info = self.curr.general_info
        w0, w1 = (
            (0, self.nwalkers)
            if walker_block is None
            else (int(walker_block[0]), int(walker_block[1]))
        )
        n_local = w1 - w0
        _gpus0 = general_info.gpus
        if _gpus0 is not None and len(_gpus0) > 1 and len(_gpus0) > n_local:
            # More devices than walker rows: a device would get an EMPTY
            # shard and the first GB gather dies on it (job 764). The layout
            # refuses this before the build (``build_layout``); this is the
            # backstop for a path that never resolved one, raised BEFORE any
            # walker container is built.
            from .communication.ranks import devices_beyond_walkers_message

            _lay = getattr(self, "layout", None)
            raise ValueError(devices_beyond_walkers_message(
                self.rank, list(_gpus0), w0, w1,
                getattr(_lay, "n_compute", 1) if _lay is not None else 1,
                self.nwalkers,
                gpu_routing=bool(getattr(_lay, "gpu_routing", False)),
            ))
        state_view = _rebuild_state_view(state, walker_block)
        coarse_stats = self._prepare_coarse_wdm_runtime(state)
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
                np.array_split(np.arange(w0, w1), len(_gpus_for_split))
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
                    # No galfor BRANCH, but a fixed foreground may still have
                    # been supplied (GALFOR_FIXED_PARAMS ->
                    # general.fixed_psd_kwargs). Falling through to None here
                    # means "no galaxy", not "no galfor branch": at 1-3 mHz the
                    # confusion is ~3x the instrument noise, so the floor
                    # collapses and the GB search births into it.
                    #
                    # This is what lets the foreground be PINNED while psd and
                    # gb stay sampled. Freezing galfor at a measured shape for
                    # a gb_search stage breaks the self-shielding loop it
                    # otherwise sits in: galfor absorbs the unresolved galaxy,
                    # which raises the search's own noise floor in exactly the
                    # band those sources live in, which stops them being
                    # found, which keeps galfor high. (Measured on the 6mo run
                    # 2026-09-23: 2x too high at 3.5 mHz, SNR x0.71, ~745
                    # findable binaries hidden.)
                    #
                    # Unset -> None -> byte-identical to the previous
                    # behaviour. Physical basis, no transform, same contract
                    # as the fixed-sensitivity path below.
                    galfor_params = (general_info.fixed_psd_kwargs or {}).get(
                        "galfor_params"
                    )
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
                coarse_stats=coarse_stats,
            )

        acs_tmp = []
        self.logger.info(
            "setup_acs: building %d walker ACs (RSS %.0f MB)",
            n_local, _rss_mb(),
        )
        for i, w in enumerate(range(w0, w1)):
            with device_context(xp, _walker_device.get(w)):
                acs_tmp.append(_build_walker_ac(w))
            if i % 8 == 7 or i == n_local - 1:
                self.logger.info(
                    "setup_acs: walker AC %d/%d built (RSS %.0f MB)",
                    i + 1, n_local, _rss_mb(),
                )

        gpus = general_info.gpus
        if gpus is not None and len(gpus) > 1 and n_local % len(gpus) != 0:
            logger.warning(
                "local rows=%d (block [%d, %d) of %d walkers) is not divisible by "
                "len(gpus)=%d: contiguous np.array_split shards are uneven, so "
                "per-shard batch sizes differ and any fixed-block intra-shard "
                "indexing is invalid (GBGPU uses rank-based indexing and stays "
                "correct). Prefer local rows %% ngpus == 0 for balanced device loads.",
                n_local, w0, w1, self.nwalkers, len(gpus),
            )
        acs = AnalysisContainerArray(
            acs_tmp,
            gpus=gpus,
            # Overlap per-split work (vectorized dispatch / signal_operation)
            # across devices; single-GPU/CPU runs stay serial.
            run_threaded=gpus is not None and len(gpus) > 1,
        )
        # The DATA the residuals are built from (a reference, not a copy):
        # the GB per-(walker, band) RJ valve judges the band gain
        # 1/2<d|d> - 1/2<r|r> (user ruling 2026-10-10), and the ACA's own
        # buffers hold only residuals.
        acs.input_data_residual_array = general_info.input_data_residual_array

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
            for i, ac in enumerate(acs.flatten()):
                # Generate + subtract this walker's templates on the device
                # that owns its shard: ``ac.data`` is a view into the ACA's
                # shard buffer (on gpus[split]), so building the template and
                # the in-place ``add_signal`` (domains.py residual add) both
                # run on that device -- no cross-device peer access.
                with device_context(xp, _walker_device.get(w0 + i)):
                    gen_map = getattr(ac, "_signal_gen", None)
                    if not isinstance(gen_map, dict):
                        continue  # this walker's branches use the fallback below
                    params = {}
                    params_pre_transformed = []
                    for name in self.curr.engine_info.branch_names:
                        if name in ("psd", "galfor") or name not in gen_map:
                            continue
                        inds_w = state_view.branches_inds[name][0, i]
                        if not inds_w.any():
                            continue
                        rows = state_view.branches_coords[name][0, i][inds_w]
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
                    get_templates(state_view, source_info, self.curr.general_info)
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

    def _global_likelihood(self, acs) -> np.ndarray:
        """The (nwalkers,) likelihood vector: local rows, allgathered across compute ranks."""
        local = np.asarray(asnumpy(acs.likelihood(complex=False)))
        fanout = getattr(self, "fanout", None)
        if fanout is None or self.layout.is_single():
            return local
        return fanout.allgather_walker_vector(local)

    @property
    def engine_info(self) -> EngineInfo:
        """EngineInfo object containing branch configuration for the sampler engine."""
        return self.curr.engine_info

    def _collect_priors_periodic(self):
        """``(priors, periodic)`` gathered from every branch's ``source_info`` entry.

        Moved verbatim out of :meth:`prepare_main` so the computation ranks
        build the identical dicts without repeating the loop.
        """
        priors = {}
        periodic = {}
        for name in self.engine_info.branch_names:
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
        return priors, periodic

    def _attach_walker_supplemental(self, state):
        """Stamp the state's ``supplemental["walker_inds"]`` (moved verbatim).

        The GLOBAL walker ids; a rank's walker slice remaps them to its own
        ``0..B-1`` (``communication.walkerslice.slice_state``).
        """
        supps_base_shape = (self.ntemps, self.nwalkers)
        walker_vals = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))
        supps = BranchSupplemental(
            {"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True
        )
        state.supplemental = supps

    def _make_fanout(self, model=None):
        """A :class:`WalkerFanout` for this compute rank (``None`` in single mode).

        Built on EVERY compute rank -- the head to drive commands, the rest for
        the symmetric setup-phase collective
        (``allgather_walker_vector``). ``model`` is attached later, once the
        rank has an engine (head) or its own RNG stream (computation ranks).

        TEST-ONLY HOOK: ``GB_PROPOSE_ORCHESTRATE=1`` also builds it in SINGLE
        mode, as a direct-call fan-out (``comm`` is ``None`` there, and
        ``WalkerFanout.run`` simply calls the body). That is the parity lever
        of ``tests/test_multirank_gb_smoke.py``: GB's ``propose`` dispatches to
        the orchestrator when a fan-out is present and the env var is set, so
        the single-rank orchestrator can be compared against ``_propose_legacy``
        in one process. ``fanout.single`` stays True, so ``fanout_active`` is
        False everywhere and every other family (addremove, PSD, the readiness
        guard, ``stop()``, ``_global_likelihood``) keeps its single-mode path.
        Never set this env var in production.
        """
        if self.layout.is_single() and os.environ.get(
            "GB_PROPOSE_ORCHESTRATE", "0"
        ) != "1":
            return None
        from .communication.fanout import WalkerFanout

        fanout = WalkerFanout(
            self.fanout_comm, self.layout, self.rank, model=model, logger=self.logger,
            group_comm=getattr(self, "group_comm", None),
            reps_comm=getattr(self, "reps_comm", None),
        )
        # The resolved base (``_resolve_seed_base``), NOT the raw config field:
        # every rank got it in the state bcast, so the clock a body reads and
        # the stream ``_seed_rank_streams`` drew agree by construction. This
        # runs AFTER that bcast on both sides (prepare_main / prepare_compute).
        fanout.clock["seed_base"] = getattr(self, "_seed_base", None)
        return fanout

    def _resolve_seed_base(self):
        """The ONE seed base every rank derives its RNG stream from (head-side).

        ``None`` in single mode: a single-process run reseeds nothing and keeps
        exactly today's stream. Multi-rank takes ``general.random_seed`` when
        the user set it and otherwise draws fresh entropy HERE, on the head
        only -- the previous ``random_seed or 0`` fallback handed every
        multi-rank run and every resubmit the identical stream (and disagreed
        with the ``None`` the fan-out clock advertised). The drawn value ships
        to the computation ranks with the state, so no rank draws its own.
        """
        if self.layout.is_single():
            return None
        seed = getattr(self.curr.general_info, "random_seed", None)
        if seed is not None:
            return int(seed)
        return int(np.random.SeedSequence().entropy % (2**32 - 1))

    def _seed_rank_streams(self):
        """Distinct, deterministic RNG streams per compute rank (multi-rank only).

        Returns the seed, or ``None`` in single mode -- where NOTHING is
        reseeded, so a single-process run keeps exactly today's stream.
        """
        if self.layout.is_single():
            return None
        from .communication.ranks import derive_rank_seed

        seed = derive_rank_seed(int(self._seed_base), self.layout, self.rank)
        np.random.seed(seed)
        if _xp_is_cupy and self.curr.general_info.gpus:
            # cupy keeps ONE RandomState PER DEVICE and ``seed()`` touches
            # only the current one. Seeded with device 0 current, a rank
            # whose walker lives on GPU 1 (ranks 2/3 of the 9mo layout) drew
            # its acceptance uniforms from a lazily created, entropy-seeded
            # device-1 state -- independent, but not reproducible from the
            # run seed. Seed the state of every device this rank owns.
            for _dev in self.curr.general_info.gpus:
                with device_context(xp, _dev):
                    xp.random.seed(seed)
        self.logger.info("rank %d RNG streams seeded with %d", self.rank, seed)
        return seed

    def _open_run_backend(self, state, priors):
        """Open/reset the run's HDF store, stamp its identities, arm mid-iteration saves.

        HEAD-ONLY: the computation ranks run an in-memory eryn backend and
        never write. Lifted verbatim out of :meth:`prepare_main`, where it
        still runs at exactly the same point -- between the initial-likelihood
        block and the recipe's ``setup_function`` -- as the
        ``after_first_likelihood`` hook of :meth:`_build_acs_and_recipe`.
        Performs NO collective. Publishes and returns ``self.run_backend``.
        """
        backend_path = self.curr.general_info.main_file_path
        general_info = self.curr.general_info
        branch_names = self.engine_info.branch_names
        ndims = self.engine_info.ndims
        nleaves_max = self.engine_info.nleaves_max
        nwalkers = general_info.nwalkers
        ntemps = general_info.ntemps

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

        _reset_backend = not backend.initialized
        if backend.initialized and int(getattr(backend, "iteration", 0) or 0) == 0:
            # An initialized store with ZERO saved iterations holds nothing
            # worth keeping but may carry dataset shapes from an OLDER
            # config: the first save_step into it would fail (or silently
            # mis-shape) at exactly the moment the run finally survives long
            # enough to save. Seen live 2026-08-27 on the 6-mo sources
            # probe: a 6-temp-era empty store outlived several 4-temp
            # relaunches because only ``not initialized`` triggered a reset.
            self.logger.info(
                "store %s is initialized but empty (0 stored iterations): "
                "re-initializing it against the current run configuration.",
                backend_path,
            )
            _reset_backend = True
        if _reset_backend:
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
        #
        # Key this on ``iteration == 0`` rather than only on the reset branch:
        # an initialized backend may be a prior attempt that died before its
        # first sample. Such an attempt can retain a stale settings block even
        # though callers correctly treat the empty chain as fresh. Conversely,
        # never replace the historical metadata of a chain that has samples;
        # doing so could erase the evidence of a mismatched resume.
        domain_settings = general_info.domain_settings
        if domain_settings is not None and int(backend.iteration) == 0:
            backend.write_domain_settings(domain_settings)

        # Noise-model identity: same shapes can hide a different likelihood
        # (unequal-arm vs equal-arm, wdm_psd_method, delay table, modulation).
        # Persist the semantic identity with a fresh chain; refuse to resume a
        # sampled chain under a different one.
        noise_identity = getattr(general_info, "noise_model_identity", None)
        _coarse_runtime = getattr(general_info, "coarse_wdm_runtime", None)
        if noise_identity and _coarse_runtime is not None:
            noise_identity = {
                **noise_identity,
                "coarse_fiducial_digest": _coarse_runtime.fiducial_digest,
            }
        # SAMPLING BASIS of the noise branches (2026-09-18). psd.log_sampling
        # and galfor.log_sampling change what the STORED NUMBERS MEAN without
        # changing a single array shape: galfor amp 2.5e-44 read back under
        # log sampling is 10**2.5e-44 ~ 1. A resume across that flip is the
        # exact "same shapes, different likelihood" case this identity exists
        # to refuse, and it was the one noise knob it did not record.
        _src = getattr(self.curr, "source_info", None) or {}
        if noise_identity:
            _basis = {}
            for _b in ("psd", "galfor"):
                _info = _src.get(_b) if hasattr(_src, "get") else None
                if _info is not None:
                    _basis[f"{_b}_log_sampling"] = bool(
                        getattr(_info, "log_sampling", False)
                    )
            if _basis:
                noise_identity = {**noise_identity, **_basis}
        if noise_identity:
            if int(backend.iteration) == 0:
                backend.write_noise_model_identity(noise_identity)
            else:
                stored = backend.read_noise_model_identity()
                if stored is None:
                    non_default = (
                        noise_identity.get("unequal_arm")
                        or noise_identity.get("galfor_modulation")
                        or noise_identity.get("wdm_psd_method", "fold") != "fold"
                    )
                    if non_default:
                        raise ValueError(
                            f"Cannot resume {backend_path!r}: it predates "
                            "noise-model identity records, but this "
                            "configuration requests a non-default noise model "
                            f"({noise_identity}). The stored chain was almost "
                            "certainly sampled under a different likelihood -- "
                            "use a fresh store (new STORE_DIR/BASE_FILE_NAME)."
                        )
                    logger.warning(
                        "Resuming %s with no stored noise-model identity "
                        "(pre-identity store); current identity is the stock "
                        "default, continuing.",
                        backend_path,
                    )
                else:
                    mismatched = {}
                    for key, value in noise_identity.items():
                        stored_value = stored.get(key)
                        # A key this store PREDATES is not a mismatch: the
                        # identity grows over time and every key added later
                        # would otherwise refuse every existing store at once
                        # (adding psd/galfor_log_sampling on 2026-09-18 would
                        # have killed both live runs on their next resume).
                        # The asymmetry is the point -- absent-and-default is
                        # fine, absent-and-NON-default is exactly the silent
                        # reinterpretation we are guarding, so that still
                        # raises below.
                        if key not in stored:
                            if value in (False, "", 0):
                                continue
                            mismatched[key] = ("<not recorded>", value)
                            continue
                        if isinstance(value, float):
                            same = np.isclose(
                                float(stored_value), value, rtol=0.0, atol=1e-6
                            )
                        else:
                            same = stored_value == value
                        if not same:
                            mismatched[key] = (stored_value, value)
                    if mismatched:
                        raise ValueError(
                            f"Cannot resume {backend_path!r}: stored "
                            "noise-model identity differs from the configured "
                            f"one: {mismatched} (stored, configured). Same "
                            "array shapes, different likelihood -- use a "
                            "fresh store or restore the original noise "
                            "configuration."
                        )

        # Arm mid-iteration checkpointing (sampling rank only -- this method
        # runs nowhere else). getattr defaults keep legacy pickled settings
        # objects working. save_step ticks the stored count via note_saved().
        if getattr(general_info, "midit_checkpoint", True):
            midit_checkpoint.arm(
                backend_path,
                min_interval=float(
                    getattr(general_info, "midit_checkpoint_min_interval", 600.0)
                ),
                stored_iteration=(
                    int(getattr(backend, "iteration", 0) or 0)
                    if backend.initialized
                    else 0
                ),
            )

        self.run_backend = backend
        return backend

    def _build_acs_and_recipe(self, state, priors, *, after_first_likelihood=None):
        """This rank's ACA block, its likelihood, and the materialized recipe.

        Shared by the head and every computation rank, and the ONLY place the
        setup-phase collectives live: :meth:`_global_likelihood` is called
        exactly TWICE -- once before the recipe's ``setup_function`` and once
        after it -- in the same order on every compute rank, so the ranks
        never desynchronize during setup.

        Args:
            state: the full (all-walker) state every rank starts from.
            priors: the prior dict from :meth:`_collect_priors_periodic`.
            after_first_likelihood: head-only hook ``fn(state, priors)`` run
                BETWEEN the two gathers, where :meth:`prepare_main` has always
                opened its HDF store -- this keeps the single-process order of
                operations byte-for-byte what it was. It must not perform any
                collective (the computation ranks do not call it).

        Returns:
            ``(acs, like_mix)`` for this rank's walker block.
        """
        # Single process: build EVERY walker exactly as before. ``None``
        # (rather than the equivalent ``(0, nwalkers)``) keeps setup_acs'
        # state view the state OBJECT itself instead of a sliced copy.
        walker_block = (
            None if self.layout.is_single() else self.layout.block_of(self.rank)
        )
        # rebuild_residuals=True: branches that registered a params-based
        # ``signal_gen`` on their Setup get their current templates
        # subtracted here, under the hood (the converted ``get_templates``
        # process). Branches without one are skipped with a warning and
        # may keep subtracting in their recipe (legacy path) -- no
        # double-subtraction either way.
        acs = self.setup_acs(
            state, rebuild_residuals=True, walker_block=walker_block
        )
        self.logger.debug("acs setup done")

        state.log_like[:] = self._global_likelihood(acs)
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

        if after_first_likelihood is not None:
            after_first_likelihood(state, priors)

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
        # Published for the builders: MoveBuildContext fills its
        # layout/fanout/rank from ``curr``, and the recipe forwards the
        # stage/iteration clock to the fan-out. Both are None in single mode.
        self.curr.fanout = self.fanout
        self.recipe.fanout = self.fanout
        # GF_RESIDUAL_SNAPSHOT_EVERY (head only; the recipe's per-iteration
        # hook is the head's): what lisatools.globalfit.residual_snapshot
        # needs to write every cold walker's residual beside the store.
        self.recipe.snapshot_context = dict(
            acs=acs,
            data_holder=getattr(self.curr.general_info,
                                "input_data_residual_array", None),
            store_path=self.curr.general_info.main_file_path,
            store_iteration0=int(getattr(self, "_resume_store_iteration", 0)),
        )
        setup_info_all = self.curr.settings_dict.setup_function(
            self.recipe, self.engine_info, self.curr, acs, priors, state
        )

        # Recipe setup can change the residual (GB_SUBTRACT_OUT_OF_BAND /
        # subtract_neighbors remove known out-of-band catalogue GBs from acs),
        # and it runs AFTER the initial-logL print above. Re-evaluate so the
        # logged value -- and the sampler's starting state.log_like -- reflect
        # the post-subtraction residual (no-op when nothing subtracted).
        state.log_like[:] = self._global_likelihood(acs)
        logger.info(f"initial log likelihood (after recipe setup): {state.log_like[0]}")

        # NULL_CHECK_ONLY: that line IS the whole measurement (per-source
        # truth-injection null test). The flag is only RAISED here -- never
        # acted on -- and :meth:`prepare_main` stops on it the moment this
        # method returns, before the sampler, plot container and checkpoint
        # self-test are built, tearing down through the NORMAL path (spares
        # released with the usual "stop" sends, compute ranks released with
        # ``fanout.stop()``, ``run_global_fit`` handing the saver its
        # {"finish_run": True}). See :func:`null_check_only`.
        #
        # DO NOT return early from HERE (dev carried the early return in
        # ``prepare_main``; this block moved into the shared setup path at
        # Plan 2). Two reasons: this method owes BOTH its callers
        # ``(acs, like_mix)``, and it runs on every COMPUTE rank as well --
        # a compute rank must fall through to ``ComputeService.serve()`` so
        # the head's fan-out STOP can release it.
        if null_check_only():
            self._null_check_only = True
            logger.info(
                "[NULL_CHECK_ONLY] initial lnL measured; skipping the "
                "sampler build and all sampling, and releasing the "
                "helper ranks."
            )

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

        # The post-recipe-setup likelihood, kept for prepare_main's third
        # likelihood site: that one is NOT a collective (the computation ranks
        # have left the setup phase by then), so multi-rank reuses this value
        # instead of recomputing a head-only partial vector.
        self._ll_after_setup = np.array(state.log_like[0], copy=True)

        # Readiness guard (n_compute > 1 only): every GlobalFitMove leaf must
        # be able to SERVE its share of the walkers, or the run would silently
        # sample only the head's block for that move.
        if not self.layout.is_single():
            unready, head_only = _fanout_unready_moves(_materialized_moves(self.recipe))
            if unready:
                raise RuntimeError(
                    f"multi-rank run with n_compute={self.layout.n_compute} but these "
                    f"moves do not serve fan-out commands yet: {unready}. Run with one "
                    "compute rank (or GF_LEGACY_RANK_LAYOUT=1) until they are ported."
                )
            if head_only:
                self.logger.warning(
                    "multi-rank run: these moves run on the HEAD ALONE (plain eryn "
                    "moves, FunctionMoves, gf_head_only): %s. A FunctionMove's fn "
                    "sees the head's walker-block ACA; a plain eryn move instead "
                    "gets the FULL ensemble state and no ACA at all, so a "
                    "zero-likelihood move (gb_ridge_gibbs, vgb_ridge_gibbs) acts on "
                    "every walker's coords -- but one that DID call the likelihood "
                    "would score all walkers against the head's block and must be "
                    "ported to the fan-out before use.",
                    head_only,
                )
        return acs, like_mix

    def prepare_main(self):
        """Build everything the HEAD needs: state, ACS, engine, recipe, backend.

        The head's half of the setup phase, in order: collect priors/periodic,
        ``load_info`` the state (this rank alone reads the store), resolve the
        run's seed base and ``bcast`` the ``(state, seed_base)`` pair to the
        computation ranks, build the fan-out, then the SHARED
        :meth:`_build_acs_and_recipe` (ACA over this rank's walker block, the
        two likelihood gathers, recipe materialization) with the head-only
        :meth:`_open_run_backend` hook running between them to open/reset the
        HDF store and arm mid-iteration saves. Then the
        :class:`GlobalFitEngine` over the GLOBAL walker count, the recipe's
        first step, this rank's RNG streams and finally the fan-out ``ping``
        handshake -- the first point-to-point traffic of the run, so it comes
        after every rank has finished its own setup.

        Single-process runs take exactly the same path with the collectives
        and the seed reseeding skipped (``layout.is_single()``). Afterwards
        ``self.sampler`` / ``self.state`` / ``self.priors`` / ``self.acs`` /
        ``self.run_backend`` / ``self.live_ctx`` are set; ``run_global_fit``
        and :meth:`sample` both start from here.
        """
        branch_names = self.engine_info.branch_names
        ndims = self.engine_info.ndims
        nleaves_max = self.engine_info.nleaves_max
        nleaves_min = self.engine_info.nleaves_min

        priors, periodic = self._collect_priors_periodic()

        state = self.load_info(priors)
        self.logger.debug("state loaded (RSS %.0f MB)", _rss_mb())

        # The run's ONE seed base, resolved before the bcast so it travels WITH
        # the state (``None`` in single mode: nothing is reseeded there).
        seed_base = self._resolve_seed_base()
        self._seed_base = seed_base

        # COLLECTIVE 1/3 (multi-rank only): the head's freshly-loaded state is
        # the one every computation rank builds its block from -- so the store
        # is read ONCE and no rank can disagree about the starting point. The
        # seed base rides along in the same payload, so one bcast serves both
        # and every rank makes the SAME number of bcast calls with the same
        # shape (prepare_compute mirrors this line). ``bcast`` returns the very
        # object we passed on the root.
        if not self.layout.is_single():
            state, self._seed_base = self.fanout_comm.bcast(
                (state, seed_base), root=self.layout.fanout_rank(self.main_rank)
            )
            self.logger.info(
                "rank layout seed base %d (set general.random_seed=%d to reproduce; "
                "the field has no env knob -- there is no RANDOM_SEED env var)",
                self._seed_base, self._seed_base,
            )

        self._attach_walker_supplemental(state)
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

        # The fan-out is built BEFORE the recipe's setup_function so every
        # move builder can read ``ctx.fanout`` / ``ctx.layout`` while it
        # materializes (MoveBuildContext fills them from ``curr``), and so
        # ``_global_likelihood`` inside _build_acs_and_recipe finds it.
        # ``None`` in single mode -- nothing below changes there.
        self.fanout = self._make_fanout(model=None)

        acs, like_mix = self._build_acs_and_recipe(
            state, priors, after_first_likelihood=self._open_run_backend
        )

        # NULL_CHECK_ONLY: the initial-lnL line _build_acs_and_recipe just
        # logged IS the whole measurement, so stop HERE -- before the
        # sampler, plot container and checkpoint self-test are built -- and
        # tear down through the NORMAL path.
        if getattr(self, "_null_check_only", False):
            # Legacy spares, waiting on a bare COMM_WORLD "stop" ...
            self._stop_spare_ranks()
            # ... and the fan-out compute ranks, which that string never
            # reaches: they are parked in ComputeService.serve() on the
            # fan-out communicator and only its STOP command releases them.
            # Without this a multi-rank NULL_CHECK_ONLY job hangs with every
            # compute rank waiting forever.
            if getattr(self, "fanout", None) is not None:
                try:
                    self.fanout.stop()
                except Exception:
                    logger.exception(
                        "fanout.stop() failed during NULL_CHECK_ONLY shutdown (ignored)"
                    )
            # run_global_fit's _null_check_only branch then hands the saver
            # its {"finish_run": True} and every rank exits.
            return

        backend = self.run_backend

        logger.debug("need to setup moves that use parallel resources")

        # backend.grow(1, None)
        # accepted = np.zeros((self.ntemps, self.nwalkers), dtype=int)
        # swaps_accepted = np.zeros((self.ntemps - 1), dtype=int)
        # backend.save_step(state, accepted, swaps_accepted=swaps_accepted)
        # exit()

        # Stop the spare processes.
        self._stop_spare_ranks()

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
        #
        # NOT a collective: compute ranks have already left the setup phase
        # by this point, so a fresh allgather here would have no partner.
        # Multi-rank reuses the post-recipe-setup gather (``_ll_after_setup``,
        # set by ``_build_acs_and_recipe`` right after that gather) instead
        # of recomputing; the getattr fallback keeps this call safe before
        # that wiring lands.
        if self.layout.is_single():
            state.log_like[:] = acs.likelihood(complex=False)[None, :]
        else:
            _ll_after_setup = getattr(self, "_ll_after_setup", None)
            if _ll_after_setup is None:
                state.log_like[:] = acs.likelihood(complex=False)[None, :]
            else:
                state.log_like[:] = _ll_after_setup[None, :]
        state.log_prior = np.zeros_like(
            state.log_like
        )  # sampler_mix.compute_log_prior(state.branches_coords, inds=state.branches_inds, supps=supps)
        self.recipe.setup_first_recipe_step(sampler_mix.iteration, state, sampler_mix)

        # Per-rank RNG streams (no-op in single mode: today's stream is kept).
        self._seed_rank_streams()

        # The head's model is the engine's -- so a fan-out command body runs
        # against the same ACA/RNG the head's own block uses. The ping is the
        # first point-to-point traffic of the run and must come AFTER the
        # computation ranks have finished their own setup (they are in, or
        # heading into, ComputeService.serve by now): it is NOT a collective,
        # so a rank still building would simply be met later.
        if self.fanout is not None:
            self.fanout.model = sampler_mix.get_model()
            self.fanout.ping()

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
        # ``state_local``: the head's OWN walker block, for a builder that must
        # act on this rank's rows only (``None`` in single mode, where the full
        # state already IS the block).
        state_local = None
        if not self.layout.is_single():
            from .communication.walkerslice import slice_state

            _w0, _w1 = self.layout.block_of(self.rank)
            state_local = slice_state(state, _w0, _w1, sub_states=[])
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
            state_local=state_local,
        )

        # Checkpoint self-test on the FULLY-BUILT state (sub-state tempered
        # blocks allocated, recipe materialized): write + read back through
        # the real resume path, so this run reports in its first minute
        # whether preemption protection actually works HERE (GPU node, this
        # MPI layout, this branch set). Never fatal.
        if midit_checkpoint.armed():
            midit_checkpoint.self_test(
                state,
                validate=self._midit_checkpoint_validate,
                logger_=self.logger,
            )

    def _stop_spare_ranks(self):
        """Release every LEGACY spare rank by sending it its ``"stop"``.

        A spare sits in ``comm.recv(source=main_rank)`` from startup (see
        ``run_global_fit``'s ``else`` branch) and exits on the first message,
        so this MUST run on every path off the main rank -- the normal
        sampling one and :func:`null_check_only`'s early return alike, which
        is why it is factored out here.

        ``self.ranks_to_give`` is the layout's ``RankRole.SPARE`` set, which
        is EMPTY in the walker-block layout; the legacy layout
        (``GF_LEGACY_RANK_LAYOUT=1``) still has one compute rank owning the
        whole pool and every other non-saver rank a stopped spare. Compute
        ranks must never appear here: they are parked in
        ``ComputeService.serve()`` on the FAN-OUT communicator and are
        released by ``fanout.stop()``, not by a bare ``"stop"`` string on
        COMM_WORLD. (The old move->rank dispatch that handed spares to moves
        was removed with the CPU distribution-fitting workers it served — GPU
        GMM fitting / neural flows replaced them; parallel-resources plan P3.
        A future coarse multi-node worker pool would re-enter here.)
        """
        for rank in self.ranks_to_give:
            self.comm.send("stop", dest=rank)

    def prepare_compute(self):
        """Build what a COMPUTATION rank needs, then hand it to the command loop.

        Mirrors :meth:`prepare_main`'s setup phase collective for collective --
        one ``bcast`` of the head's state, then the two gathers inside
        :meth:`_build_acs_and_recipe` -- and nothing else, so both roles reach
        the sampling phase in lockstep. What differs: this rank's ACA holds
        only its walker block, its eryn engine is an in-memory shell (no HDF
        store, no plots, no stopping function) that exists so the recipe steps
        can stamp ``periodic`` / ``temperature_control`` onto the moves, and it
        never proposes anything itself -- the head drives every move through
        :class:`~lisatools.globalfit.communication.fanout.ComputeService`.

        Sets ``self.compute_service`` (plus ``self.sampler`` / ``self.state`` /
        ``self.priors`` / ``self.acs`` for symmetry with the head).
        """
        priors, periodic = self._collect_priors_periodic()

        # COLLECTIVE 1/3: the head's state and the run's resolved seed base
        # (it alone reads the store and it alone draws the base). Same call
        # shape as the head's line in prepare_main -- one bcast of a
        # ``(state, seed_base)`` pair, so neither side can drift in call count.
        state, self._seed_base = self.fanout_comm.bcast(
            None, root=self.layout.fanout_rank(self.main_rank)
        )
        self._attach_walker_supplemental(state)
        self.fanout = self._make_fanout(model=None)

        # COLLECTIVES 2/3 and 3/3 live in here (the two likelihood gathers).
        acs, like_mix = self._build_acs_and_recipe(state, priors)

        from eryn.moves import StretchMove
        from eryn.utils import PeriodicContainer

        periodic_key_order = {key: value.key_order for key, value in priors.items()}
        if periodic and not isinstance(periodic, PeriodicContainer):
            periodic = PeriodicContainer(periodic, key_order=periodic_key_order)

        # The shell engine stays at the GLOBAL nwalkers (ruling): RecipeStep
        # .setup_run stamps sampler.temperature_control onto every move that
        # lacks one, so N keeps this rank symmetric with the head's engine.
        engine = GlobalFitEngine(
            acs,
            self.nwalkers,
            self.engine_info.ndims,
            like_mix,
            priors,
            tempering_kwargs={"ntemps": self.ntemps},
            nbranches=len(self.engine_info.branch_names),
            nleaves_max=self.engine_info.nleaves_max,
            nleaves_min=self.engine_info.nleaves_min,
            moves=StretchMove(live_dangerously=True),
            rj_moves=None,
            kwargs=None,
            # in-memory eryn Backend: never written, never read by the saver.
            backend=None,
            vectorize=True,
            periodic=periodic,
            branch_names=self.engine_info.branch_names,
            plot_generator=None,
            plot_iterations=-1,
            provide_groups=True,
            provide_supplemental=True,
            track_moves=False,
            stopping_fn=None,
        )

        # Stamp every step's moves once (periodic / temperature_control /
        # thinning). The ONE tolerated failure is a legacy step reading sampler
        # state this shell engine does not carry (AttributeError); the head
        # re-stamps the stage kind on every command anyway
        # (ComputeService.handle). Anything else is a real defect and
        # propagates to the abort hook -- a rank that silently serves
        # half-stamped moves would desync from the head.
        for step in self.recipe.recipe:
            name = step.get("name") if isinstance(step, dict) else None
            adjust = step.get("adjust") if isinstance(step, dict) else step
            try:
                adjust.setup_run(0, state, engine)
            except AttributeError as exc:
                self.logger.warning(
                    "rank %d: setup_run stamping for recipe step %r read sampler state the "
                    "compute shell engine lacks (%s); continuing.",
                    self.rank, name, exc,
                )

        seed = self._seed_rank_streams()
        rank_rng = np.random.RandomState(seed)
        model = GlobalFitInfo(acs, map, rank_rng)
        self.fanout.model = model

        # Addressable by (stage, name) -- keyed from the STEP LIST, because a
        # stock runtime move is ONE object shared by every stage that lists it
        # and its ``gf_stage_name`` stamp keeps only the last stage
        # (see _serve_registry). Exactly the leaves the head's readiness
        # guard vetted, plus an unambiguous bare-name fallback.
        registry = _serve_registry(self.recipe)
        stage_keys = sorted(k for k in registry if isinstance(k, tuple))
        self.logger.info(
            "rank %d serving %d (stage, move) pair(s): %s",
            self.rank, len(stage_keys), [f"{s}/{n}" for s, n in stage_keys],
        )

        from .communication.fanout import (
            LIKELIHOOD_OP,
            RESIDUAL_HASH_OP,
            ComputeService,
            residual_hash,
        )
        from .residual_snapshot import RESIDUAL_SNAPSHOT_OP, walker_residual

        self.compute_service = ComputeService(
            self.fanout_comm,
            self.layout,
            self.rank,
            registry=registry,
            model=model,
            builtins={
                # answers WalkerFanout.gather_likelihood (head-side "all
                # walkers" residual reads during sampling, e.g. FunctionMove)
                LIKELIHOOD_OP: lambda payload, clock, model: np.asarray(
                    asnumpy(model.analysis_container_arr.likelihood(complex=False))
                ),
                # answers WalkerFanout.gather_residual_hashes (one-walker
                # replica-mode agreement check on the [FANOUT_DIGEST] line)
                RESIDUAL_HASH_OP: lambda payload, clock, model: residual_hash(
                    model.analysis_container_arr
                ),
                # answers residual_snapshot.take_snapshot: a block's lead
                # returns its walkers' residuals (GF_RESIDUAL_SNAPSHOT_EVERY)
                RESIDUAL_SNAPSHOT_OP: lambda payload, clock, model: walker_residual(
                    model, payload
                ),
            },
            logger=self.logger,
        )
        self.sampler = engine
        self.state = state
        self.priors = priors
        self.acs = acs

    def run_global_fit(self):
        """Execute the run for this rank's role (head / compute / saver / legacy spare)."""
        backend_path = self.curr.general_info.main_file_path
        if self.role == RankRole.HEAD:
            self.prepare_main()

            if getattr(self, "_null_check_only", False):
                # NULL_CHECK_ONLY: prepare_main stopped after the initial-lnL
                # print and has already released the spares AND the fan-out
                # compute ranks. There is no sampler and no stored iteration,
                # so there is nothing to sample and nothing to write a
                # submission from -- fall straight through to the saver's
                # finish_run below so every rank exits now.
                logger.info(
                    "[NULL_CHECK_ONLY] no sampling; finishing the run."
                )
            else:
                try:
                    self.sampler.run_mcmc(
                        self.state, self.curr.general_info.num_iterations, thin_by=1,
                        progress=self.progress, store=True,
                    )
                    self._write_submission()
                    logger.info("Residuals saved.")
                finally:
                    # Every compute rank is parked in ``ComputeService.serve()``
                    # waiting for the next command; only STOP releases it. On the
                    # happy path this is the ordinary shutdown, and on a head
                    # exception it is what lets the workers exit instead of
                    # blocking forever (under real MPI the abort hook ends the
                    # job either way -- this makes a FakeWorld / in-process run,
                    # where there is no abort, terminate cleanly too).
                    if getattr(self, "fanout", None) is not None:
                        try:
                            self.fanout.stop()
                        except Exception:
                            # A failing stop() must never supplant an exception already
                            # propagating out of run_mcmc (Python `finally` semantics:
                            # an exception raised here would otherwise replace it).
                            logger.exception("fanout.stop() failed during shutdown (ignored)")


            if self.results_rank != self.main_rank:
                self.comm.send({"finish_run": True}, dest=self.results_rank)
        elif self.role == RankRole.SAVER:
            backend = GFHDFBackend(
                backend_path,
                sub_backend=self.engine_info.branch_backends,
                sub_state_bases=self.engine_info.branch_states,
            )
            plot_container = self.make_plot_container()
            self._release_rank_gpu_pool()
            # GB warm-start SELF refit builder (GB_WARM_SELF=1; None = off):
            # runs on THIS rank after its own saves, on this rank's device
            from .warmstart.self_refit import make_warm_self_hook

            save_to_backend_asynchronously_and_plot(
                backend, self.comm, self.main_rank,
                plot_container=plot_container, plot_iter=self._plot_iterations,
                backup_iter=self.curr.general_info.backup_iter,
                warm_self_hook=make_warm_self_hook(
                    self.curr, store_path=backend_path),
            )
        elif self.role == RankRole.COMPUTE:
            self.prepare_compute()
            served = self.compute_service.serve()
            #: how many commands this rank answered (read by the launch smoke)
            self.compute_service_served = served
            self._release_rank_gpu_pool()
            self.logger.info("compute rank %d served %d command(s); exiting.", self.rank, served)
        else:  # legacy SPARE: wait for the startup "stop" and exit
            self._release_rank_gpu_pool()
            info = self.comm.recv(source=self.main_rank)
            logger.info(f"Process {self.rank} finished ({info!r}).")

    def _write_submission(self):
        """End-of-run submission dump (head-only, single-rank-only).

        Reads the HEAD's ACA, which in a multi-rank run holds just its walker
        block: ``postprocessing.save_residuals`` would write
        ``residual_0..residual_{B-1}`` under full-run names and
        ``_prepare_gb_samples`` would argmax over that block alone, so the
        dump has to be routed through the fan-out before it can run there.
        """
        if not self.layout.is_single():
            self.logger.warning(
                "multi-rank run: submission residual dump SKIPPED -- the head's ACA holds only "
                "walkers [%d, %d) of %d; TODO(multi-rank): route SubmissionWriter through the "
                "fan-out (gather residuals per block) before enabling it.",
                *self.layout.block_of(self.rank), self.layout.nwalkers,
            )
            return
        if self.curr.general_info.submission_parent_folder is not None:
            self.logger.debug(
                f"saving submission to {self.curr.general_info.submission_parent_folder}"
            )
            submission_writer = SubmissionWriter(
                backend=self.run_backend, curr=self.curr, ess=20_000
            )
            submission_writer.write_submission(self.acs)

    def _release_rank_gpu_pool(self):
        """Release this rank's build-time GPU memory cache on its OWN device(s).

        Under multi-rank launches every rank runs the full ``build()`` (data
        load, WDM transforms, F-stat staging) before the roles resolve, so
        the saver and (legacy) spare ranks each sit on device memory they
        will never use again -- measured via the gpu_procs telemetry on the
        2026-08-22 production jobs: 3.4 GB/rank at 3 months, 4.6 GB/rank at 1
        year, parked on ONE device. Both v5 crashes that night were
        allocation failures on that same device at ~97-99% -- the helpers'
        cache was the missing margin. Frees CACHED pool blocks (after a
        ``gc.collect()``; live arrays are untouched, so this is
        behavior-neutral).

        Iterates ``self.curr.general_info.gpus`` (the rank-local device list
        AFTER ``communication.ranks.select_rank_device`` pinning), NOT
        ``layout.local_gpus(rank)``: the layout's pool ids are the PER-NODE
        pool's own numbering, which in ``"visible"`` pinning mode no longer
        even exist as device indices (the rank's ``CUDA_VISIBLE_DEVICES`` was
        narrowed to just its own devices, renumbered ``0..k-1``). In the legacy
        layout (``GF_LEGACY_RANK_LAYOUT=1``) ``prepare_rank`` ->
        ``select_rank_device`` still runs: it leaves the COMPUTE rank's whole
        pool untouched (mode ``"legacy"``, ``ranks.py`` ~386-387) but narrows
        the saver/spare ranks' ``general.gpus`` to the single device their
        placement names (``ranks.py`` ~276-279) -- which is exactly the device
        those ranks built on. ``general_info.gpus`` is therefore correct in
        every mode: ``[0..k-1]`` in ``"visible"`` mode, the pinned pool ids in
        ``"setdevice"`` mode, the untouched full pool for a legacy compute rank
        and the single placed device for a legacy saver/spare. A CPU run
        (``gpus`` empty/``None``) is a no-op.
        """
        import gc

        gc.collect()
        try:
            import cupy as cp
        except Exception:
            return
        devices = list(self.curr.general_info.gpus or [])
        if not devices:
            return
        freed = 0
        for dev in devices:
            try:
                with cp.cuda.Device(int(dev)):
                    pool = cp.get_default_memory_pool()
                    freed += pool.total_bytes() - pool.used_bytes()
                    pool.free_all_blocks()
            except Exception:
                continue
        self.logger.info(
            "rank %d released ~%.2f GB of cached GPU pool blocks on device(s) %s.",
            self.rank, freed / 1e9, list(devices),
        )

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
