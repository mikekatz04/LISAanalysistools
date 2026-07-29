"""Move that swaps individual sources in/out of the residual data array."""

from __future__ import annotations

import inspect
import logging
import os
import time
from copy import deepcopy
from typing import Any, Callable, Optional, TYPE_CHECKING

try:
    import cupy as xp
    _xp_is_cupy = True
except (ImportError, ModuleNotFoundError):
    # CPU-only fallback so the moves package imports cleanly when cupy is
    # unavailable (e.g. on a Mac development host).
    import numpy as xp
    _xp_is_cupy = False
import numpy as np


def _free_pool() -> None:
    """No-op stand-in for ``cupy.get_default_memory_pool().free_all_blocks()`` on CPU."""
    if _xp_is_cupy:
        xp.get_default_memory_pool().free_all_blocks()


from eryn.moves import Move, StretchMove, TemperatureControl
from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer

from tqdm import tqdm

from ...analysiscontainer import AnalysisContainerArray
from ...domaincomputation import DomainComputationGroupArray
from ...domains import DomainBase, DomainBaseArray
from ...utils.utility import asnumpy, get_array_module
from .globalfitmove import GlobalFitMove

logger = logging.getLogger(__name__)
DEBUG_MODE = False

if TYPE_CHECKING:
    from ...sources.waveformbase import TDWaveformBase
    from ...domaincomputation import DomainComputationGroupArray


class MoveSignalGen:
    """Engine-convention params-based generator adapter for one move.

    Follows the same call convention as the engine-installed per-branch
    generators (e.g. the stock ``SourceSignalGen``):
    ``fn(*sampling_params, apply_transform=True, **waveform_kwargs) ->
    DomainBase``, with the branch transform applied inside by default and
    ``apply_transform=False`` meaning the caller already applied it (the
    add/remove move always calls this way — its choreography transforms once
    up front, so the transform is applied exactly once either way).

    Registered into a container's dict-based ``signal_gen`` under the move's
    branch name when the container does not already carry a usable generator
    there. Reads ``transform_fn`` / ``waveform_gen`` off the move at call
    time so later reassignment (e.g. the multi-GPU subclass swapping
    ``waveform_gen`` after the base ``__init__``) stays honored; a specific
    ``waveform_gen`` override (heterodyne hook) can be pinned instead.
    """

    def __init__(self, move: "ResidualAddOneRemoveOneMove", waveform_gen: Callable = None):
        self.move = move
        self._waveform_gen = waveform_gen

    @property
    def waveform_gen(self) -> Callable:
        return self._waveform_gen if self._waveform_gen is not None else self.move.waveform_gen

    def __call__(self, *params, apply_transform: bool = True, leaf_inds=None, **kwargs):
        if apply_transform:
            params = self.move.transform_fn.both_transforms(
                np.asarray(params, dtype=float), leaf_inds=leaf_inds
            )
        return self.waveform_gen(*params, **kwargs)


class ResidualAddOneRemoveOneMove(GlobalFitMove, StretchMove, Move):
    """
    Move that handles adding and removing sources to and from the residuals stored in the analysis container array.
    This is done by first removing the contribution of the current sources in the cold chain from the residual,
    then proposing new sources for this leaf, and then adding back in the contribution of the new sources to the residual.
    This way we can make sure that the likelihoods are computed correctly for each proposed source and that the likelihoods are consistent with the current state of the residuals in the analysis container array.

    Args:
        branch_name: name of the branch that this move will operate on.
        coords_shape: shape of the coordinates of the sources in the branch that this move will operate on.
        waveform_gen: function that generates the waveforms for the sources given their (waveform-basis) coordinates. All operations on the analysis container array run through the containers' ``signal_gen`` machinery (:meth:`AnalysisContainer.build_template` / :meth:`AnalysisContainer.calculate_signal_likelihood`) and default to the generator the engine already installed on each container under ``branch_name``; ``waveform_gen`` is the fallback, wrapped as a :class:`MoveSignalGen` and added to containers that do not carry the branch (see :meth:`_resolve_signal_gen_override`). Because the move transforms coordinates once up front, the containers' generators are always called with ``apply_transform=False`` — the transform is applied exactly once on every path.
        waveform_gen_kwargs: keyword arguments for the waveform generator function.
        waveform_like_kwargs: keyword arguments for the likelihood computation function.
        acs: analysis container array that contains the residuals and other information needed for the likelihood computation.
        num_repeats: number of times to repeat the proposal step for each leaf.
        transform_fn: transform container that contains the transforms to be applied to the coordinates before generating waveforms and computing likelihoods.
        priors: prior distribution container that contains the prior distributions for the sources in the branch.
        inner_moves: list of moves and their corresponding weights to be used for proposing new sources for the leaf.
        Tmax: maximum temperature for the temperature control.
        betas_all: array of betas for all leaves and temperatures. Shape is (nleaves_max, ntemps). If None, betas will be initialized as in TemperatureControl.
        permute_every: gate for the walker-permuting (fancy) temperature swap: > 0 enables it — it fires exactly ONCE per leaf visit, on the final in-model repeat — and <= 0 disables it entirely (``{BRANCH}_PERMUTE_EVERY`` env override). The numeric value is no longer a cadence.
        pad_out_of_prior: whether to pad proposed sources that are out of the prior bounds to avoid JIT compilation issues. If True, proposed sources that are out of the prior bounds will be replaced with the first in-prior point. 
        **kwargs: additional keyword arguments for the Move class.
    """

    def __init__(
        self,
        branch_name: str,
        coords_shape: tuple,
        waveform_gen: Callable,
        waveform_gen_kwargs: dict,
        waveform_like_kwargs: dict,
        acs: AnalysisContainerArray,
        num_repeats: int,
        transform_fn: TransformContainer,
        priors: ProbDistContainer,
        inner_moves: list,
        Tmax: float = np.inf,
        betas_all: np.ndarray = None,
        permute_every: int = 20,
        pad_out_of_prior: bool = False,
        dcga: DomainComputationGroupArray = None,
        waveform_gen_method: str = None,
        waveform_like_method: str = None,
        run_async: bool = False,
        run_threaded: bool = False,
        **kwargs,
    ):

        # Move.__init__(self, **kwargs)
        StretchMove.__init__(self, **kwargs)

        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = coords_shape

        # ONE move structure (2026-07 multi-GPU pass): pass ``dcga=`` to run
        # waveform generation + likelihood scoring on per-device replicas —
        # how many GPUs the move uses is set by the DCGA it is built with.
        # Without a DCGA the move runs the per-container ACA path (which is
        # itself shard-aware; see compute_acs_like). Same object either way.
        # With ``dcga``, ``waveform_gen`` is the generator OBJECT (its
        # ``.kwargs`` seed the replicas) and ``waveform_gen_method`` names
        # the method; without, ``waveform_gen`` is the bound callable.
        self._dcga = dcga
        self._run_async = run_async
        self._run_threaded = run_threaded
        self._waveform_gen_obj = None
        self.waveform_gen_method = waveform_gen_method
        self.waveform_like_method = waveform_like_method or waveform_gen_method
        if dcga is not None:
            if acs is None:
                acs = dcga.acs
            if waveform_gen_method is not None:
                self._waveform_gen_obj = waveform_gen
                waveform_gen = getattr(waveform_gen, waveform_gen_method)

        self.branch_name = branch_name
        self.acs = acs
        self.waveform_gen = waveform_gen
        self.num_repeats = num_repeats
        self.transform_fn = transform_fn
        self.priors = priors
        self.waveform_gen_kwargs = waveform_gen_kwargs
        self.waveform_like_kwargs = waveform_like_kwargs
        moves_tmp = [move[0] if isinstance(move, tuple) else move for move in inner_moves]
        move_weights = [move[1] if isinstance(move, tuple) else 1.0 for move in inner_moves]
        self.moves = moves_tmp
        self.move_weights = move_weights / np.sum(move_weights)

        self.temperature_controls = [None for _ in range(self.nleaves_max)]
        for i in range(self.nleaves_max):
            if betas_all is not None:
                assert betas_all.shape == (self.nleaves_max, self.ntemps)
                betas_in = betas_all[i]
            else:
                betas_in = None

            self.temperature_controls[i] = TemperatureControl(
                self.ndim,
                self.nwalkers,
                betas=betas_in,
                permute=False,
                ntemps=self.ntemps,
                Tmax=Tmax,
                skip_swap_branches=None,  # will fill in after first run through move
            )
        
        self.permute_every = permute_every
        self.pad_out_of_prior = pad_out_of_prior

        self._setup_debug()

        # make sure to propagate the periodic information to the inner moves if it is included in kwargs
        if 'periodic' in kwargs:
            self.periodic = kwargs['periodic']

        if self._dcga is not None and self._waveform_gen_obj is not None:
            self.create_waveform_gen_replicas()

    # ------------------------------------------------------------------
    # DCGA (multi-device) spread — absorbed from the legacy
    # MultiGPUResidualAddRemoveMove in the 2026-07 multi-GPU pass (one
    # move structure; active when the move is constructed with ``dcga=``).
    # ------------------------------------------------------------------

    @property
    def dcga(self) -> DomainComputationGroupArray:
        return self._dcga

    @property
    def run_async(self) -> bool:
        return self._run_async

    @property
    def run_threaded(self) -> bool:
        return self._run_threaded

    def create_waveform_gen_replicas(self):
        """One waveform generator per device, seeded from the generator
        object's ``.kwargs`` with the split's device-local orbits."""
        gen_obj = self._waveform_gen_obj
        if not hasattr(gen_obj, "kwargs"):
            raise ValueError(
                "Waveform generator must have a 'kwargs' attribute that "
                "contains the keyword arguments to initialize the waveform "
                "generator (needed to build per-device replicas)."
            )
        self._waveform_generators = []
        for i, device in enumerate(
            self.dcga.gpus
            if self.dcga.gpus is not None
            else [None] * self.dcga.num_splits
        ):
            with self.dcga.device_context(device):
                init_kwargs = gen_obj.kwargs.copy()
                if "orbits" in init_kwargs:
                    init_kwargs["orbits"] = (
                        self.dcga.computation_groups[i].orbits
                    )
                self._waveform_generators.append(
                    gen_obj.__class__(**init_kwargs)
                )

    @property
    def waveform_generators(self) -> list:
        return self._waveform_generators

    def make_args_tuple(self, coords) -> tuple:
        """Unpack coordinates into per-parameter arrays for the generator
        (JAX needs the separate memory addresses)."""
        return tuple(xp.array(coords[:, i]) for i in range(coords.shape[1]))

    def prepare_inputs(self, coords, data_index):
        """Partition coordinates/indices by split and place them on device."""
        positions_per_split, data_intra_index_per_split, _ = (
            self.dcga.unpack_indices(data_index)
        )
        coords_per_split = self.dcga.unpack_coords(
            positions_per_split, coords, keep_tuple=True
        )

        data_intra_index_per_split, coords_per_split = (
            self.dcga.place_on_device(
                items=(data_intra_index_per_split, coords_per_split)
            )
        )

        waveform_args_per_split = self.dcga._loop_operation(
            operation=[
                self.make_args_tuple for _ in self.dcga.computation_groups
            ],
            operation_args_per_split=coords_per_split,
        )

        return (
            positions_per_split,
            data_intra_index_per_split,
            waveform_args_per_split,
        )

    def aggregate_waveforms(self, waveforms_per_split) -> list[DomainBase]:
        """Flatten per-split waveform lists into one list."""
        waveforms = []
        for waveforms_split in waveforms_per_split:
            waveforms.extend(waveforms_split)
        return waveforms

    def _get_waveform_here_dcga(self, coords: np.ndarray) -> list[DomainBase]:
        self.free_gpu_memory()

        data_index = np.arange(coords.shape[0], dtype=np.int32)

        _, _, waveform_args_per_split = self.prepare_inputs(coords, data_index)

        operations = [
            getattr(waveform_gen, self.waveform_gen_method)
            for waveform_gen in self.waveform_generators
        ]

        waveforms_out = self.dcga._loop_operation(
            operation=operations,
            operation_args_per_split=waveform_args_per_split,
            operation_kwargs=self.waveform_gen_kwargs,
            aggregate_fn=self.aggregate_waveforms,
            run_threaded=self.run_threaded,
        )

        self.dcga.synchronize()

        return waveforms_out

    def _compute_like_dcga(
        self, coords_in: np.ndarray, data_index: np.ndarray
    ) -> np.ndarray:
        (
            positions_per_split,
            data_intra_index_per_split,
            waveform_args_per_split,
        ) = self.prepare_inputs(coords_in, data_index)

        waveform_like_operations = [
            getattr(waveform_gen, self.waveform_like_method)
            for waveform_gen in self.waveform_generators
        ]

        likelihood_args_per_split = self.dcga._loop_operation(
            operation=waveform_like_operations,
            operation_args_per_split=waveform_args_per_split,
            operation_kwargs=self.waveform_like_kwargs,
            positions_per_split=positions_per_split,
            run_threaded=self.run_threaded,
        )

        if not self.run_async:
            self.dcga.synchronize()

        likelihoods = self.dcga.compute_signal_likelihood(
            positions_per_split=positions_per_split,
            data_intra_per_split=data_intra_index_per_split,
            noise_intra_per_split=data_intra_index_per_split,
            likelihood_args_per_split=likelihood_args_per_split,
            likelihood_kwargs={'run_async': self.run_async},
            run_threaded=self.run_threaded,
        )

        # Release GPU arrays before freeing the pool: the large template
        # arrays must be dereferenced before free_all_blocks() so those
        # blocks are actually returned rather than staying pool-owned.
        del (
            likelihood_args_per_split,
            waveform_args_per_split,
            data_intra_index_per_split,
        )

        self.free_gpu_memory()

        if np.any(~np.isfinite(likelihoods)):
            logger.warning(
                f"Non-finite likelihoods encountered: {likelihoods}. "
                "This could be a sign of numerical issues."
            )
            if DEBUG_MODE:
                breakpoint()

        likelihoods = np.where(
            np.isfinite(likelihoods), likelihoods, -1e300
        )

        return likelihoods

    # ------------------------------------------------------------------
    # Debug / instrumentation plotting (mirrors the GB special-stretch move,
    # adapted to the "remove source from residual -> sample -> put it back"
    # choreography of this move). Activated per branch by the capitalised
    # ``{BRANCH_NAME}_DEBUG`` env var (e.g. EMRI_DEBUG / SOBBH_DEBUG /
    # MBH_DEBUG); companion knobs mirror GB's:
    #   {BRANCH}_DEBUG            "1" arms all debug hooks (default off)
    #   {BRANCH}_DEBUG_DIR        output folder (default ./gf_output/{branch}_debug/)
    #   {BRANCH}_DEBUG_PLOT_WALKER the single walker whose leaf is plotted (0)
    #   {BRANCH}_DEBUG_PLOT_LEAF   the single leaf plotted (unset -> the first
    #                              alive leaf each step)
    #   {BRANCH}_DEBUG_EVERY       plot only every Nth propose step (1)
    # ------------------------------------------------------------------
    def _setup_debug(self) -> None:
        """Read the per-branch ``{BRANCH}_DEBUG`` env options (called from __init__)."""
        p = str(self.branch_name).upper()
        self._dbg_prefix = p
        self.debug = bool(int(os.environ.get(f"{p}_DEBUG", "0")))
        self.debug_plot_dir = os.environ.get(
            f"{p}_DEBUG_DIR", f"./gf_output/{self.branch_name}_debug/"
        )
        self.debug_plot_walker = int(os.environ.get(f"{p}_DEBUG_PLOT_WALKER", "0"))
        _leaf = os.environ.get(f"{p}_DEBUG_PLOT_LEAF")
        self.debug_plot_leaf = int(_leaf) if _leaf not in (None, "") else None
        self.debug_every = max(1, int(os.environ.get(f"{p}_DEBUG_EVERY", "1")))
        # prev_logl consistency check (DEFAULT ON): cross-check the move's
        # scoring path (compute_like — the DCGA/GPU path when active) against
        # the per-container ACS path at the current-state points each leaf
        # visit. "0" disables, "1" warns (default), "strict" raises. Costs one
        # extra ntemps*nwalkers likelihood batch per leaf per propose.
        self.check_ll_mode = (
            os.environ.get(f"{p}_CHECK_LL")
            or os.environ.get("ADDREMOVE_CHECK_LL", "1")
        ).strip().lower()
        self.check_ll_every = max(
            1, int(os.environ.get(f"{p}_CHECK_LL_EVERY", "1"))
        )
        # Tempering diagnostics: {BRANCH}_SWAP_DEBUG=1 logs one [SWAP_DBG]
        # line per repeat (cold-chain lnL before/after the temperature swap,
        # inner acceptance, per-rung swap counts, fancy flag).
        self.swap_debug = bool(int(os.environ.get(f"{p}_SWAP_DEBUG", "0")))
        # {BRANCH}_PERMUTE_EVERY overrides the ctor permute_every; <= 0
        # DISABLES the walker-permuting (fancy) swap entirely — the A/B lever
        # for isolating fancy-swap pathologies.
        _pe_env = os.environ.get(f"{p}_PERMUTE_EVERY") or os.environ.get(
            "ADDREMOVE_PERMUTE_EVERY"
        )
        if _pe_env is not None:
            self.permute_every = int(_pe_env)
            logger.info(
                "[%s] permute_every overridden via env -> %d (<=0 disables "
                "the fancy swap)", p, self.permute_every,
            )
        # Colour scale keyed to the RESIDUAL (where the other sources are
        # already subtracted) and log10 by default, so this source stays
        # visible even in a crowded band. {P}_DEBUG_LOG=0 -> linear.
        self.debug_log = bool(int(os.environ.get(f"{p}_DEBUG_LOG", "1")))
        self._dbg_plot_counter = 0
        self._dbg_step = 0
        if self.debug:
            logger.info(
                "[%s_DEBUG] armed: dir=%s walker=%d leaf=%s every=%d",
                p, self.debug_plot_dir, self.debug_plot_walker,
                self.debug_plot_leaf, self.debug_every,
            )

    def _dbg_trace_this_step(self) -> bool:
        """True when the current propose step should produce debug plots."""
        return self.debug and (self._dbg_step % self.debug_every == 0)

    def _dbg_residual(self, walker: int):
        """Host copy of ``walker``'s current residual array (domain-native shape)."""
        return asnumpy(self.acs[int(walker)].data.arr).copy()

    def _dbg_template(self, coords_row_in, walker: int):
        """Host copy of one source's template array, in the residual's domain shape.

        ``coords_row_in`` is a single already-transformed (waveform-basis)
        coordinate row. Built through the walker's container ``signal_gen``
        machinery — the same path :meth:`_apply_cold_chain_sources` folds into
        the residual — so the plotted template matches the fit exactly.
        Returns ``None`` on any failure (debug must never break the sampler).
        """
        try:
            ac = self.acs[int(walker)]
            sig = ac.build_template(
                {self.branch_name: np.asarray(coords_row_in)},
                waveform_kwargs=self._branch_waveform_kwargs(),
                signal_gen=self._resolve_signal_gen_override(ac),
                apply_transform=False,
            )
            arr = sig.arr if hasattr(sig, "arr") else sig
            return asnumpy(arr).copy()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s_DEBUG] template build skipped: %r", self._dbg_prefix, exc)
            return None

    def _dbg_weighted_ll(self, residual_arr, walker) -> float:
        """``-1/2 <r|r>`` for a residual snapshot, PSD-weighted.

        Never hand-roll the inner product: build the snapshot as the domain
        object the walker's container expects and let the container's OWN
        installed :meth:`AnalysisContainer.inner_product` apply the sensitivity
        weighting (the same primitive the sampler scores with). A raw
        ``sum|r|^2`` drops the PSD entirely and is not a likelihood.
        """
        ac = self.acs[int(walker)]
        settings = ac.data.settings
        snapshot = settings.associated_class(asnumpy(residual_arr), settings)
        saved = ac.data
        try:
            ac.data = snapshot
            return -0.5 * float(np.real(ac.inner_product()))
        finally:
            ac.data = saved

    def _debug_plot_source_sequence(self, leaf, walker, snaps, template, meta) -> None:
        """Save a GB-style flip-book of ``[template | data | residual]`` frames.

        Mirrors the GB special-stretch move's per-moment 3x3 sequence figures:
        one figure PER MOMENT of the remove -> sample -> re-add choreography,
        each with rows = TDI channels (X/Y/Z) and columns

          1. TOTAL TEMPLATE — this leaf's recovered source ``h`` (fixed),
          2. TOTAL DATA     — the data this source is fit against, i.e. the
                              source-isolated residual with every other source
                              already removed (fixed reference),
          3. RESIDUAL       — the ACS residual AT THAT MOMENT (varies).

        The frames are the entry (source folded into the fit), removed (source
        isolated for sampling), and refit (updated source folded back) states.
        Template and data columns are held fixed across frames while the
        residual column changes, so flipping through the numbered files
        ``..._f0_...`` / ``_f1_`` / ``_f2_`` animates the source leaving and
        re-entering the residual directly. A single colour scale (per channel)
        is shared across ALL frames and columns so the flip is calibrated.
        Domain-adaptive (WDM image / FD|TD line); any failure is logged and
        swallowed.

        NOTE vs GB: this move does not mutate the residual during the repeat
        loop (proposals are scored on the fly), so the GB "before/after
        add-back" pair collapses to a single ``removed`` frame here.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if template is None:
                return
            tmpl = np.asarray(template)
            # Fixed "data" reference = the source-isolated residual (others
            # removed). Fall back to entry + template if the exposed snapshot
            # is missing.
            data = snaps.get("exposed")
            if data is None and snaps.get("before_removal") is not None:
                data = np.asarray(snaps["before_removal"]) + tmpl
            if data is None:
                return
            data = np.asarray(data)
            if tmpl.shape != data.shape:
                return  # template build failed / shape mismatch -> skip cleanly

            frames = [
                ("f0", "entry: source in fit", snaps.get("before_removal")),
                ("f1", "removed: source isolated", snaps.get("exposed")),
                ("f2", "refit: source re-added", snaps.get("after_readd")),
            ]
            frames = [(k, t, np.asarray(r)) for k, t, r in frames if r is not None]
            if not frames:
                return

            nch = data.shape[0] if data.ndim >= 2 else 1
            chan_labels = (
                ["X", "Y", "Z"] if nch == 3 else [f"chan {c}" for c in range(nch)]
            )
            kind = meta.get("domain", "wdm")

            def _ch(a, c):
                # Signed real coefficients (WDM w_mn are real), so a diverging
                # colour map centres an emptied template / clean residual at
                # white and shows signal as colour.
                a = np.real(np.asarray(a))
                a = a[None] if a.ndim == 1 else a
                return a[c] if a.shape[0] > c else a[0]

            # Diverging colour scale (the WDM class default, RdBu), symmetric
            # about zero and keyed to the RESIDUAL frames only (the other
            # sources are already subtracted there, so a crowded band does not
            # blow out the scale). SymLogNorm (the ``debug_log`` default) keeps
            # a weak source visible against a large dynamic range; the
            # template/data columns share the same norm and simply saturate
            # where they are loud. One norm per channel, shared across all
            # frames -> the flip-book is calibrated.
            from matplotlib.colors import Normalize, SymLogNorm

            res_pool = [
                np.concatenate([_ch(r, ch).ravel() for _, _, r in frames])
                for ch in range(nch)
            ]
            use_log = bool(getattr(self, "debug_log", True)) and kind == "wdm"
            norm_row = []
            for ch in range(nch):
                a = np.abs(res_pool[ch])
                vmax = float(np.percentile(a[a > 0], 99.5)) if np.any(a > 0) else 1.0
                vmax = vmax or 1.0
                norm_row.append(
                    SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax, base=10)
                    if use_log else Normalize(vmin=-vmax, vmax=vmax)
                )

            rr_entry = (
                self._dbg_weighted_ll(snaps["before_removal"], walker)
                if snaps.get("before_removal") is not None else float("nan")
            )
            os.makedirs(self.debug_plot_dir, exist_ok=True)
            base = f"{self.branch_name}_debug_leaf{leaf}_w{walker}_s{self._dbg_step:04d}"

            for fkey, ftitle, residual in frames:
                # The "total template" is the CURRENT total template = data -
                # residual. When the source is removed (f1) it leaves the model
                # and reappears in the residual, so with a single source its
                # power switches from the template column to the residual
                # column across the flip-book.
                total_template = np.asarray(data) - np.asarray(residual)
                cols = [("total template", total_template), ("total data", data),
                        (f"residual — {ftitle}", residual)]
                fig, axes = plt.subplots(
                    nch, 3, figsize=(3.6 * 3, 2.7 * nch), squeeze=False, sharex=True,
                )
                for ci, (ctitle, a) in enumerate(cols):
                    for ch in range(nch):
                        ax = axes[ch][ci]
                        slab = _ch(a, ch)
                        if kind == "wdm" and slab.ndim == 2:
                            im = ax.imshow(
                                slab, aspect="auto", origin="lower",
                                cmap="RdBu", norm=norm_row[ch],
                            )
                            if ci == 2:
                                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                                cb.set_label("coeff", fontsize=7)
                        else:
                            ax.plot(_ch(a, ch).ravel(), lw=0.6)
                        if ch == 0:
                            ax.set_title(ctitle, fontsize=9)
                        if ci == 0:
                            ax.set_ylabel(chan_labels[ch], fontsize=9)
                rr_here = self._dbg_weighted_ll(residual, walker)
                fig.suptitle(
                    f"{self.branch_name} leaf {leaf} walker {walker} step "
                    f"{self._dbg_step} — {ftitle}  | -0.5<r|r>_psd entry={rr_entry:.3e} "
                    f"here={rr_here:.3e}",
                    fontsize=10,
                )
                fname = os.path.join(self.debug_plot_dir, f"{base}_{fkey}.png")
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(fname, dpi=120, bbox_inches="tight")
                plt.close(fig)
                logger.info("[%s_DEBUG] saved -> %s", self._dbg_prefix, fname)
            self._dbg_plot_counter += 1
        except Exception as exc:  # noqa: BLE001
            logger.info("[%s_DEBUG] plot skipped: %r", self._dbg_prefix, exc)

    @property
    def periodic(self):
        """Periodicity dictionary propagated to all inner moves."""
        return self._periodic

    @periodic.setter
    def periodic(self, periodic):
        self._periodic = periodic
        if periodic is not None:
            for tmp_move in self.moves:
                if tmp_move.periodic is None:
                    tmp_move.periodic = periodic

    def free_gpu_memory(self):
        if self._dcga is not None:
            self._dcga.free_gpu_memory()
            return
        if self.xp is not np:
            self.xp.get_default_memory_pool().free_all_blocks()

    def check_add_skip_swap_info(self, state):
        """Populate ``skip_swap_branches`` on each per-leaf temperature control.

        Branches other than :attr:`branch_name` are skipped during temperature
        swapping so this move does not corrupt unrelated branches.
        """

        if self.temperature_controls[0].skip_swap_branches is not None:
            return

        if len(state.branches) > 1:
            skip_swap_branches = [key for key in state.branches.keys()]
            skip_swap_branches.remove(self.branch_name)

        else:
            skip_swap_branches = []

        for i in range(self.nleaves_max):
            self.temperature_controls[i].skip_swap_branches = skip_swap_branches

    @property
    def own_signal_gen(self) -> MoveSignalGen:
        """This move's engine-convention generator adapter (lazy singleton)."""
        if getattr(self, "_own_signal_gen", None) is None:
            self._own_signal_gen = MoveSignalGen(self)
        return self._own_signal_gen

    def _supports_apply_transform(self, gen) -> bool:
        """``True`` when ``gen`` explicitly accepts the ``apply_transform`` kwarg.

        The move feeds waveform-basis parameters, so a generator is only
        usable as the branch default if it can be told to skip its internal
        sampling→waveform transform (a bare ``**kwargs`` doesn't count — it
        would silently forward the flag into the wave wrap). Results are
        cached per generator object.
        """
        cache = getattr(self, "_apply_transform_support_cache", None)
        if cache is None:
            cache = self._apply_transform_support_cache = {}
        key = id(gen)
        if key not in cache:
            try:
                parameters = inspect.signature(gen).parameters
                cache[key] = any(
                    p.name == "apply_transform"
                    and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in parameters.values()
                )
            except (TypeError, ValueError):
                cache[key] = False
        return cache[key]

    def _resolve_signal_gen_override(self, ac):
        """Resolve the branch generator for one container: installed first.

        Default (returns ``None``): the generator already installed on the
        container's dict-based ``signal_gen`` under :attr:`branch_name` —
        the same generator the engine uses for residual rebuilds — provided
        it supports the ``apply_transform`` kwarg the move relies on.

        When the container does not carry that model name, this move's
        :attr:`own_signal_gen` is added: durably registered when that cannot
        clobber anything (no generator installed, or a dict without the
        key), otherwise (bare single-callable ``signal_gen``, or an
        installed entry without ``apply_transform`` support) injected for
        the call only via the returned per-call ``{branch: gen}`` mapping.
        """
        gen_map = getattr(ac, "_signal_gen", None)
        if isinstance(gen_map, dict):
            installed = gen_map.get(self.branch_name)
            if installed is not None:
                if self._supports_apply_transform(installed):
                    return None
                # installed under our name but cannot skip its transform:
                # score/apply with our own generator, without clobbering it.
                return {self.branch_name: self.own_signal_gen}
            new_map = dict(gen_map)
            new_map[self.branch_name] = self.own_signal_gen
            ac.signal_gen = new_map
            return None
        if gen_map is None:
            ac.signal_gen = {self.branch_name: self.own_signal_gen}
            return None
        # bare single-callable installed by someone else: never clobber it.
        return {self.branch_name: self.own_signal_gen}

    def _branch_waveform_kwargs(self) -> dict:
        """Per-model ``waveform_kwargs`` mapping for this branch's generator."""
        return {self.branch_name: self.waveform_gen_kwargs}

    def _apply_cold_chain_sources(self, coords, sign):
        """One-at-a-time template build + apply to keep peak RAM flat.

        ``sign=-1`` subtracts (add_back), ``sign=+1`` adds (remove). ``coords``
        are already-transformed (waveform-basis) rows, one per walker. Each
        walker's template is assembled by its container's ``signal_gen``
        machinery (:meth:`AnalysisContainer.build_template`), defaulting to
        the generator the engine installed for this branch — with
        ``apply_transform=False`` since the move already applied the branch
        transform (applied exactly once either way) — and falling back to
        this move's own :class:`MoveSignalGen` when the container doesn't
        carry the branch (see :meth:`_resolve_signal_gen_override`). The
        template is then applied with the device-aware
        :meth:`AnalysisContainerArray.signal_operation`. Never holds more
        than a single source's waveform in memory.
        """
        # Delegated to the ACA (2026-07-28): apply_signal_from_params is the
        # shard-aware fused build+apply primitive (one live template per
        # split, per-split device contexts, gc/pool hygiene) — the move no
        # longer reimplements the ACA's residual machinery. A current-state
        # source outside the waveform's domain of validity is skipped
        # ("skip"): deterministic, so the add and remove passes skip
        # identically — its template is zero, matching its -1e300 sentinel.
        self.acs.apply_signal_from_params(
            sign,
            {self.branch_name: coords},
            index=np.arange(int(np.shape(coords)[0])),
            waveform_kwargs=self._branch_waveform_kwargs(),
            signal_gen_resolver=self._resolve_signal_gen_override,
            apply_transform=False,
            domain_error="skip",
        )

    def add_back_in_cold_chain_sources(self, coords):
        """Subtract current cold-chain sources from the residual (one walker at a time)."""
        self._apply_cold_chain_sources(coords, sign=-1)

    def remove_cold_chain_sources(self, coords):
        """Add current cold-chain sources back into the residual (one walker at a time)."""
        self._apply_cold_chain_sources(coords, sign=+1)

        # ll_tmp3 = self.acs.likelihood(
        #     source_only=True
        # )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

    def get_waveform_here(self, coords: np.ndarray) -> DomainBaseArray | list[DomainBase]:
        """Get the waveforms for the given source coordinates.

        Each call to ``waveform_gen`` returns a :class:`~lisatools.domains.DomainBase`
        (e.g. :class:`~lisatools.domains.STFTSignal` or
        :class:`~lisatools.domains.FDSignal`).  The results are collected into a
        :class:`~lisatools.domains.DomainBaseArray`, which batches them for
        vectorized downstream operations when all signals share the same
        domain settings.

        Args:
            coords: Source coordinates, shape ``(n_sources, ndim)``.

        Returns:
            :class:`~lisatools.domains.DomainBaseArray` of length ``n_sources``.

        """
        if self._dcga is not None:
            return self._get_waveform_here_dcga(coords)

        _free_pool()

        waveforms = []
        for i in range(coords.shape[0]):
            waveforms.append(self.waveform_gen(*coords[i], **self.waveform_gen_kwargs))

        return DomainBaseArray(waveforms)

    def setup_likelihood_here(self, coords):
        """Set up state before computing the likelihood (DCGA path computes
        the per-split <d|d> terms; the plain path needs nothing)."""
        if self._dcga is not None:
            self._dcga.compute_d_d_terms()

    def compute_acs_like(self, coords_in, data_index, signal_gen=None, **kwargs):
        """
        Compute the likelihood for the given coordinates and data index using the analysis container array.

        Args:
            coords_in: already-transformed (waveform-basis) coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim). The containers' generators are called with ``apply_transform=False`` so the branch transform — applied once by this move's choreography — is never applied twice.
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).
            signal_gen: optional waveform generator override. By default (``None``) each targeted container's own installed generator for this branch is used (falling back to this move's generator when the container doesn't carry it — see :meth:`_resolve_signal_gen_override`), so scoring and residual bookkeeping share one template-assembly path. Pass a waveform-basis callable to score with a different generator than the one used for proposing, for example when using heterodyned likelihoods; it is wrapped in :class:`MoveSignalGen` and swapped in per call.
            kwargs: additional keyword arguments for the likelihood computation function.

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """
        # TODO: we should probably move the prior in here even though
        # in general with current setup it should only be points in the prior
        # that make it here
        # Normalize to a 2D ``(n_sources, ndim)`` batch: ``atleast_2d`` on a
        # 1D vector yields ``(1, ndim)`` (row), which is exactly one source --
        # NOT ndim sources.
        coords_in = get_array_module(coords_in).atleast_2d(coords_in)
        data_index_np = np.atleast_1d(asnumpy(data_index)).astype(int)
        source_only = kwargs.pop("source_only", False)

        # Params-based likelihood through each targeted container's
        # signal_gen machinery (template assembly + scoring live in
        # :meth:`AnalysisContainer.calculate_signal_likelihood`; the move no
        # longer generates waveforms itself). Rows are evaluated one at a
        # time, which is required anyway: every generator currently wired
        # into this move is SINGLE-SOURCE (EMRI FEW GenerateEMRIWaveform;
        # the SOBBH / MBH tdionfly + legacy ResponseWrapper wraps all build
        # ONE combined template per call), so a batched
        # ``signal_gen(*coords_in.T)`` would crash or silently sum candidates.
        custom_override = (
            None if signal_gen is None
            else {self.branch_name: MoveSignalGen(self, signal_gen)}
        )

        # Delegated to the ACA (2026-07-28): scoring goes through the ACA's
        # vectorized dispatcher (calculate_signal_likelihood), which owns the
        # shard grouping, per-split device contexts, and the -1e300
        # WaveformDomainError sentinel — the move no longer reimplements the
        # sharded scoring loop. The per-container generator override is
        # resolved up front (this also durably registers the move's generator
        # on containers that don't carry the branch); rows are grouped by
        # resolved override so each group is ONE vectorized call.
        n_rows = len(data_index_np)
        if custom_override is not None:
            groups = [(custom_override, np.arange(n_rows))]
        else:
            def _ov_key(ov):
                return (
                    None if ov is None
                    else tuple(sorted((k, id(v)) for k, v in ov.items()))
                )

            by_key: dict = {}
            for pos, i in enumerate(data_index_np):
                ov = self._resolve_signal_gen_override(self.acs[int(i)])
                key = _ov_key(ov)
                if key not in by_key:
                    by_key[key] = (ov, [])
                by_key[key][1].append(pos)
            groups = [
                (ov, np.asarray(rows, dtype=int))
                for ov, rows in by_key.values()
            ]

        ll = np.full(n_rows, -1e300, dtype=float)
        for ov, rows in groups:
            vals = self.acs.calculate_signal_likelihood(
                {self.branch_name: coords_in[rows]},
                index=data_index_np[rows],
                domain_error_value=-1e300,
                signal_gen=ov,
                waveform_kwargs=self._branch_waveform_kwargs(),
                apply_transform=False,
                source_only=source_only,
                **kwargs,
            )
            ll[rows] = np.real(np.asarray(vals))
        return ll

    def compute_like(self, coords_in, data_index):
        """
        Compute the likelihood for the given coordinates and data index.

        Defaults to each container's installed generator for this branch
        (see :meth:`compute_acs_like`), so proposal scoring uses the same
        template-assembly path as the residual bookkeeping.

        Args:
            coords_in: already-transformed (waveform-basis) coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim).
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """
        if self._dcga is not None:
            return self._compute_like_dcga(coords_in, data_index)
        return self.compute_acs_like(
            coords_in, data_index, **self.waveform_like_kwargs
        )

    def setup(self, model, state):
        """Per-iteration setup hook (no-op by default)."""
        return

    def _verify_entry_vs_acs(self, prev_logl, cold_ref, leaf):
        """The expose invariant (2026-07-28): the cold-chain ``prev_logl``
        scored against the freshly EXPOSED residual must reproduce the full
        ACS likelihoods from just before the expose — algebraically
        ``r_exposed - h_current == r_full``, so the two are the same number.
        A large per-walker SPREAD in the difference means the expose/fold
        choreography (or the scoring path) is inconsistent with the residual
        state — this is the check that catches the d-2h expose-sign class of
        bug that corrupted all add/remove sampling from 6d33265 (2026-05-22)
        until today. A constant offset with ~zero spread is a benign
        normalization convention. Shares {BRANCH}_CHECK_LL gating/severity.
        """
        if cold_ref is None:
            return
        cold = np.asarray(prev_logl[0], dtype=float)
        ref = np.asarray(cold_ref, dtype=float).reshape(cold.shape)
        both = (
            np.isfinite(cold) & np.isfinite(ref)
            & (cold > -1e299) & (ref > -1e299)
        )
        if not np.any(both):
            return
        diff = cold[both] - ref[both]
        spread = float(diff.max() - diff.min())
        med = float(np.median(diff))
        if spread <= 1e-1 and abs(med) <= 5.0:
            return
        msg = (
            f"{self.branch_name} leaf {leaf}: EXPOSE INVARIANT VIOLATED — "
            f"cold prev_logl vs pre-expose ACS lnL: median offset "
            f"{med:.6e}, spread {spread:.6e} over {int(both.sum())} walkers. "
            "Large spread/offset means the exposed residual or scoring path "
            "is wrong (e.g. template subtracted instead of added at expose)."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)
        if DEBUG_MODE:
            breakpoint()

    def _verify_prev_logl(self, prev_logl, old_coords_in, data_index_in, leaf):
        """Cross-check the move's scoring path against the ACS container path.

        ``prev_logl`` came from :meth:`compute_like` — the DCGA (multi-device)
        path when active, else the container path with the engine-installed
        generator and ``waveform_like_kwargs``. This recomputes the SAME
        current-state points through :meth:`compute_acs_like` with the move's
        OWN generator and ``source_only=True`` (the resurrected intent of the
        original commented validation). Any bookkeeping drift between the
        residual state and the scoring path — the failure mode that silently
        walks a chain away from truth — shows up here as a per-point
        difference. Default mode warns loudly; ``{BRANCH}_CHECK_LL=strict``
        raises; ``0`` disables; ``{BRANCH}_CHECK_LL_EVERY=N`` thins the cost.

        A CONSTANT per-walker offset (identical across temperatures) can also
        arise from a benign ``source_only`` convention mismatch — the warning
        distinguishes the two by reporting the spread of the difference in
        addition to its maximum.
        """
        like_kwargs = {
            k: v for k, v in self.waveform_like_kwargs.items() if k != "source_only"
        }
        acs_like = (
            self.compute_acs_like(
                old_coords_in,
                data_index=data_index_in,
                signal_gen=self.waveform_gen,
                source_only=True,
                **like_kwargs,
            )
            .reshape(prev_logl.shape)
            .real
        )
        both = (
            np.isfinite(prev_logl)
            & np.isfinite(acs_like)
            & (prev_logl > -1e299)
            & (acs_like > -1e299)
        )
        if not np.any(both):
            return
        diff = prev_logl[both] - acs_like[both]
        max_abs = float(np.abs(diff).max())
        if max_abs <= 1e-1:
            return
        spread = float(diff.max() - diff.min())
        msg = (
            f"{self.branch_name} leaf {leaf}: prev_logl (move scoring path) vs "
            f"ACS container path disagree: max|diff|={max_abs:.6e}, "
            f"spread={spread:.6e} over {int(both.sum())} points. A large SPREAD "
            "means residual/scoring bookkeeping drift (real bug); a constant "
            "offset (spread ~ 0) can be a benign source_only convention "
            "difference."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)
        if DEBUG_MODE:
            breakpoint()

    #: Leaf currently being processed by the per-leaf proposal loop. The
    #: transform helper below reads it because some transform call sites
    #: (the tempering likelihood callback) have an eryn-fixed signature and
    #: cannot receive leaf indices as arguments.
    _current_leaf: Optional[int] = None

    def _to_phys(self, coords):
        """Sampling -> physical rows through the branch transform.

        For containers with PER-LEAF fills (Eryn per-leaf ``fill_dict``,
        e.g. the EMRI xI0/Phi_theta0 prograde-retrograde flags), every row
        handled by this move belongs to the leaf currently being processed
        (``self._current_leaf``); scalar-fill containers (SOBBH/MBH) take
        the plain path.
        """
        if getattr(self.transform_fn, "n_leaf_fills", None) is not None:
            assert self._current_leaf is not None, (
                "per-leaf transform used outside the per-leaf loop"
            )
            leaf_inds = np.full(
                np.shape(coords)[:-1], int(self._current_leaf), dtype=int
            )
            return self.transform_fn.both_transforms(coords, leaf_inds=leaf_inds)
        return self.transform_fn.both_transforms(coords)

    def log_like_for_fancy_swaping(self, x, supps=None, branch_supps=None, **kwargs):
        """
        Compute the log likelihood for the given coordinates and data index for use in fancy swapping.
        This is needed because when permuting the coordinates during tempering, we need to recompute the likelihood against the new set of residuals and covariance matrix.

        Args:
            x: Dictionary of coordinates of the sources for which we want to compute the likelihood.
                The coordinates are expected to be in the shape (ntemps, nwalkers, nleaves_max, ndim).
            supps: supplimental information for the likelihood computation. #todo add
            branch_supps: Branch supplimental. #todo add

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (ntemps, nwalkers).
            blobs: blobs for the given coordinates and data index. Default is None.
        """
        assert x[self.branch_name].ndim == 4 and x[self.branch_name].shape[1] == self.nwalkers
        # Single-leaf slices only: rows are scored per walker, and with
        # per-leaf transform fills the fixed ``_current_leaf`` identity must
        # apply to every row (a multi-leaf slice would silently apply one
        # leaf's fills to all leaves).
        assert x[self.branch_name].shape[2] == 1, (
            "log_like_for_fancy_swaping expects a single-leaf coordinate slice"
        )
        # shape is (nwalkers, 1 (nleaves_max), ndim)
        ntemps = x[self.branch_name].shape[0]

        coords = x[self.branch_name].reshape(-1, x[self.branch_name].shape[-1])
        data_index_in = np.tile(np.arange(self.nwalkers), (ntemps, 1)).flatten().astype(np.int32)

        coords_in = self._to_phys(coords)

        # TODO: need to be careful here when heterodyning about if it is "close"
        output = (
            self.compute_like(
                coords_in,
                data_index=data_index_in,
            )
            .reshape((ntemps, self.nwalkers))
            .real
        )
        return output, None  # AS: match psd? I'm not sure

    def get_split_inds(self):
        all_inds = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))
        inds = all_inds % self.nsplits
        if self.randomize_split:
            [np.random.shuffle(x) for x in inds]

        return inds

    def propose(self, model, state):
        """Generate proposals by removing, updating, and re-adding each leaf.

        # TODO/DOCS: detailed acceptance/swap accounting; see code for the per-leaf
        loop and the ``compute_like`` calls used to score proposals.

        Args:
            model: ``eryn`` model object containing ``analysis_container_arr``,
                ``random``, etc.
            state: Current sampler state.

        Returns:
            Tuple ``(new_state, accepted)``.
        """
        self.setup(model, state)
        tic = time.time()

        if not np.any(state.branches[self.branch_name].inds):
            ntemps, nwalkers = state.branches[self.branch_name].shape[:2]
            _accepted = np.zeros((ntemps, nwalkers), dtype=bool)
            return state, _accepted

        new_state = deepcopy(state)

        # per-step debug reset (trace at most one leaf per propose call)
        self._dbg_step += 1
        self._dbg_leaf_done = False
        self._dbg_snaps = {}

        # self.acs = model.analysis_container_arr
        self.check_add_skip_swap_info(state)

        # mapping information
        temp_inds_base = np.repeat(np.arange(self.ntemps)[:, None], self.nwalkers, axis=-1)
        walker_inds_base = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))

        # randomize order
        leaves_random_order = np.random.permutation(np.arange(self.nleaves_max))
        for leaf in leaves_random_order:
            # guard against leaves with False
            assert np.all(
                state.branches[self.branch_name].inds[0, 0, leaf]
                == state.branches[self.branch_name].inds[:, :, leaf]
            )
            if not state.branches[self.branch_name].inds[0, 0, leaf]:
                continue
            # leaf identity for the per-leaf transform fills (see _to_phys)
            self._current_leaf = int(leaf)
            # second step of randomizing order (making sure it does not run over)

            # fill this temperature control with temperatures from current state
            temperature_control_here = self.temperature_controls[leaf]

            temperature_control_here.betas[:] = new_state.sub_states[self.branch_name].betas_all[
                leaf
            ][
                : self.ntemps
            ]  # as: make sure only local ntemps are used
            ntemps_full = new_state.sub_states[self.branch_name].betas_all[leaf].shape[0]

            ndim = new_state.branches[self.branch_name].coords.shape[-1]

            # Debug: trace ONE leaf per step (the configured leaf, else the
            # first alive one). Snapshot the residual the sampler will score
            # against, before/after isolating this leaf, and plot it.
            _dbg_leaf = (
                self.debug
                and not self._dbg_leaf_done
                and self._dbg_step % self.debug_every == 0
                and (self.debug_plot_leaf is None or leaf == self.debug_plot_leaf)
            )
            _dbg_w = min(int(self.debug_plot_walker), self.nwalkers - 1)
            # Debug must never crash the sampler: every snapshot is guarded
            # and disarms this leaf's tracing on any failure.
            if _dbg_leaf:
                try:
                    self._dbg_snaps = {"before_removal": self._dbg_residual(_dbg_w)}
                    self._dbg_domain = type(self.acs[_dbg_w].data).__name__.replace(
                        "Signal", ""
                    ).lower()
                except Exception as exc:  # noqa: BLE001
                    logger.info("[%s_DEBUG] snapshot skipped: %r", self._dbg_prefix, exc)
                    _dbg_leaf = False

            # Cold-chain ACS reference BEFORE the expose: consumed by the
            # [STAGE] print and by the default-on expose invariant
            # (_verify_entry_vs_acs) — scoring the current coords against the
            # exposed residual must reproduce these values.
            _cold_ref = None
            if self.swap_debug or (
                self.check_ll_mode != "0"
                and self._dbg_step % self.check_ll_every == 0
            ):
                _cold_ref = asnumpy(self.acs.likelihood())
            if self.swap_debug:
                logger.info(
                    "[STAGE] %s leaf %d PRE-EXPOSE  ACS cold lnL min/med/max "
                    "%.3f / %.3f / %.3f",
                    self.branch_name, leaf,
                    float(np.min(_cold_ref)),
                    float(np.median(_cold_ref)),
                    float(np.max(_cold_ref)),
                )

            # remove cold chain sources FROM THE FIT: add their templates back
            # into the residual (r = d - h  ->  r = d), exposing this leaf.
            # NOTE 2026-07-28: propose() had these two calls SWAPPED since the
            # 6d33265 sign refactor (2026-05-22) — the expose step SUBTRACTED
            # the template (r -> d - 2h), so every proposal was scored against
            # a corrupted target (prev_logl ~ -2<d|d> at truth) and accepted
            # updates folded back against it degraded the true residual.
            removal_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            removal_coords_in = self._to_phys(removal_coords)
            self.remove_cold_chain_sources(removal_coords_in)

            if _dbg_leaf:
                # source-isolated residual = the "data" this source is fit
                # against (all other sources removed); the fixed reference for
                # the flip-book decomposition.
                try:
                    self._dbg_snaps["exposed"] = self._dbg_residual(_dbg_w)
                except Exception as exc:  # noqa: BLE001
                    logger.info("[%s_DEBUG] snapshot skipped: %r", self._dbg_prefix, exc)
                    _dbg_leaf = False

            self.setup_likelihood_here(removal_coords_in)

            old_coords = (
                new_state.branches[self.branch_name]
                .coords[: self.ntemps, :, leaf]
                .reshape(-1, ndim)
            )
            old_coords_in = self._to_phys(old_coords)

            data_index_in = (
                np.tile(np.arange(self.nwalkers), (self.ntemps, 1)).flatten().astype(np.int32)
            )
            # TODO: fix this
            # prev_logl = self.waveform_gen.get_direct_ll(fd, data_residuals.flatten(), psd.flatten(), self.df, *old_coords_in.T, noise_index=noise_index, data_index=data_index, **self.waveform_kwargs).reshape((ntemps, nwalkers)).real.get()
            # TODO: check if psd term is included properly here at each step
            # TODO: and check data index here
            prev_logl = (
                self.compute_like(
                    old_coords_in,
                    data_index=data_index_in,
                )
                .reshape((self.ntemps, self.nwalkers))
                .real
            )

            if self.check_ll_mode != "0" and (
                self._dbg_step % self.check_ll_every == 0
            ):
                self._verify_prev_logl(prev_logl, old_coords_in, data_index_in, leaf)
                self._verify_entry_vs_acs(prev_logl, _cold_ref, leaf)

            if self.swap_debug:
                logger.info(
                    "[SWAP_DBG] %s leaf %d ENTRY cold prev_logl min/med/max "
                    "%.3f / %.3f / %.3f",
                    self.branch_name, leaf,
                    float(prev_logl[0].min()),
                    float(np.median(prev_logl[0])),
                    float(prev_logl[0].max()),
                )

            # -1e300 is the out-of-domain sentinel, not a numerical issue.
            if np.any((prev_logl < -1e10) & (prev_logl > -1e299)) or np.any(prev_logl > 1e30):
                logger.warning(f"Very low log likelihood encountered in propose: min = {prev_logl.min()}, max = {prev_logl.max()}. This could be a sign of numerical issues.")
                if DEBUG_MODE:
                    breakpoint()

            prev_logp = (
                self.priors[self.branch_name]
                .logpdf(old_coords)
                .reshape((self.ntemps, self.nwalkers))
            )

            prev_logP = temperature_control_here.compute_log_posterior_tempered(
                prev_logl, prev_logp
            )

            # fix this need to compute prev_logl for all walkers
            _free_pool()
            for repeat in tqdm(
                range(self.num_repeats), desc=f"{self.branch_name} update, leaf {leaf}",
                disable=not getattr(self, "progress", False),
            ):

                # pick move
                move_here = self.moves[
                    model.random.choice(np.arange(len(self.moves)), p=self.move_weights)
                ]

                # logger.debug(f"move here: {move_here.__class__.__name__}")

                # Split the ensemble in half and iterate over these two halves.
                accepted = np.zeros((ntemps_full, self.nwalkers), dtype=bool)
                inds = self.get_split_inds()

                # prepare accepted fraction
                # accepted_here = np.zeros((self.ntemps, self.nwalkers), dtype=bool)
                for split in range(self.nsplits):
                    # get split information
                    S1 = inds == split
                    num_total_here = np.sum(inds == split)
                    nwalkers_here = np.sum(S1[0])

                    temp_inds_here = temp_inds_base[inds == split]
                    walker_inds_here = walker_inds_base[inds == split]

                    # prepare the sets for each model
                    # goes into the proposal as (ntemps * (nwalkers / subset size), nleaves_max, ndim)
                    sets = [
                        new_state.branches[self.branch_name]
                        .coords[: self.ntemps][inds == j][:, leaf]
                        .reshape(self.ntemps, -1, 1, ndim)
                        for j in range(self.nsplits)
                    ]

                    old_points = sets[split].reshape((self.ntemps, nwalkers_here, ndim))

                    # setup s and c based on splits
                    s = {self.branch_name: sets[split]}
                    c = {self.branch_name: sets[:split] + sets[split + 1 :]}

                    # Get the move-specific proposal.
                    if isinstance(move_here, StretchMove):
                        q, factors = move_here.get_proposal(s, c, model.random)

                    else:
                        q, factors = move_here.get_proposal(s, model.random)

                    new_points = q[self.branch_name].reshape((self.ntemps, nwalkers_here, ndim))

                    # Compute prior of the proposed position
                    # new_inds_prior is adjusted if product-space is used
                    logp = self.priors[self.branch_name].logpdf(new_points.reshape(-1, ndim))
                    in_prior = ~np.isinf(logp)
                    logl = np.full_like(logp, -1e300)

                    if np.any(in_prior):
                        if self.pad_out_of_prior and np.any(~in_prior):
                            padded = new_points.reshape(-1, ndim).copy()
                            padded[~in_prior] = new_points.reshape(-1, ndim)[in_prior][0]
                            new_points_in = self._to_phys(padded)

                            data_index = np.asarray(walker_inds_here.astype(np.int32))

                            all_logl = self.compute_like(new_points_in, data_index=data_index)
                            logl = np.where(in_prior, all_logl, -1e300)

                        else:
                            new_points_in = self._to_phys(
                                new_points.reshape(-1, ndim)[in_prior]
                            )

                            # Compute the lnprobs of the proposed position.
                            data_index = np.asarray(walker_inds_here[in_prior].astype(np.int32))
                
                            logl[in_prior] = self.compute_like(
                                    new_points_in,
                                    data_index=data_index,
                                )
                    
                    if DEBUG_MODE:
                        logger.debug(f"average proposed logl: {logl[in_prior].mean()}.")
    
                    # -1e300 is the out-of-domain sentinel, not a numerical issue.
                    _logl_in = logl[in_prior]
                    if np.any((_logl_in < -1e10) & (_logl_in > -1e299)) or np.any(_logl_in > 1e30):
                        logger.warning(f"Suspicious likelihood encountered in propose: min = {logl[~np.isinf(logp)].min()}, max = {logl[~np.isinf(logp)].max()}. This could be a sign of numerical issues.")
                        if DEBUG_MODE:
                            breakpoint()
                    # print(f"new logl: {logl}. elapsed: {time.time() - tic}")

                    logl = logl.reshape(self.ntemps, nwalkers_here)

                    logp = logp.reshape(self.ntemps, nwalkers_here)
                    prev_logp_here = prev_logp[inds == split].reshape(self.ntemps, nwalkers_here)

                    prev_logl_here = prev_logl[inds == split].reshape(self.ntemps, nwalkers_here)

                    prev_logP_here = temperature_control_here.compute_log_posterior_tempered(
                        prev_logl_here, prev_logp_here
                    )
                    logP = temperature_control_here.compute_log_posterior_tempered(logl, logp)

                    lnpdiff = factors + logP - prev_logP_here

                    keep = lnpdiff > np.log(model.random.rand(self.ntemps, nwalkers_here))

                    temp_inds_update = temp_inds_here[keep.flatten()]
                    walker_inds_update = walker_inds_here[keep.flatten()]

                    accepted[: self.ntemps][(temp_inds_update, walker_inds_update)] = True

                    # update state information
                    new_state.branches[self.branch_name].coords[
                        (
                            temp_inds_update,
                            walker_inds_update,
                            np.full_like(walker_inds_update, leaf),
                        )
                    ] = new_points[keep].reshape(len(temp_inds_update), ndim)

                    prev_logl[(temp_inds_update, walker_inds_update)] = logl[keep].flatten()
                    prev_logp[(temp_inds_update, walker_inds_update)] = logp[keep].flatten()
                    prev_logP[(temp_inds_update, walker_inds_update)] = logP[keep].flatten()

                # acceptance tracking
                self.accepted += accepted
                # print(self.accepted[0])
                self.num_proposals += 1

                # TODO: include PSD likelihood in swaps?
                # temperature swaps
                # make swaps
                coords_for_swap = {
                    self.branch_name: new_state.branches_coords[self.branch_name][
                        :, :, leaf
                    ].copy()[:, :, None]
                }

                # Walker-permuting (fancy) swap: exactly ONCE per leaf visit,
                # on the FINAL in-model repeat — the in-model stretch repeats
                # run undisturbed, then the ensemble permutes across the
                # ladder before the updated sources fold back. permute_every
                # is now a pure gate: > 0 enables, <= 0 disables
                # ({BRANCH}_PERMUTE_EVERY env override).
                fancy_swap = (
                    self.permute_every > 0 and repeat == self.num_repeats - 1
                )
                if fancy_swap:
                    logger.debug(
                        "%s leaf %d repeat %d: fancy (walker-permuting) "
                        "temperature swap firing",
                        self.branch_name, leaf, repeat,
                    )
                #if fancy_swap:
                    # logger.debug(f"Permuting walkers before swap.")
                compute_log_like = self.log_like_for_fancy_swaping

                if self.swap_debug:
                    _cold_before = np.array(prev_logl[0], copy=True)

                # TODO: check permute make sure it is okay
                (
                    coords_for_swap,
                    prev_logP,
                    prev_logl,
                    prev_logp,
                    inds,
                    blobs,
                    supps,
                    branch_supps,
                ) = temperature_control_here.temperature_swaps(
                    coords_for_swap,
                    prev_logP.copy(),
                    prev_logl.copy(),
                    prev_logp.copy(),
                    branch_supps={self.branch_name: None},  # TODO: adjust this to be flexible
                    fancy_swap=fancy_swap,
                    compute_log_like=compute_log_like,
                    permute_here=fancy_swap,
                )

                if self.swap_debug:
                    _cold_after = np.asarray(prev_logl[0])
                    _drop = _cold_before - _cold_after
                    logger.info(
                        "[SWAP_DBG] %s leaf %d rep %d fancy=%d | inner cold "
                        "acc %d/%d | cold lnL max/med %.3f / %.3f -> %.3f / "
                        "%.3f | worst walker drop %.3f | swaps_acc %s",
                        self.branch_name, leaf, repeat, int(fancy_swap),
                        int(np.asarray(accepted[0]).sum()), self.nwalkers,
                        float(_cold_before.max()),
                        float(np.median(_cold_before)),
                        float(_cold_after.max()),
                        float(np.median(_cold_after)),
                        float(_drop.max()),
                        np.array2string(
                            np.asarray(
                                temperature_control_here.swaps_accepted
                            ),
                            precision=0,
                        ),
                    )

                temperature_control_here.adapt_temps()

                new_state.branches_coords[self.branch_name][:, :, leaf] = coords_for_swap[
                    self.branch_name
                ][:, :, 0]

            # ll_tmp1 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

            # add back cold chain sources
            _free_pool()

            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self._to_phys(add_coords)

            _dbg_tmpl = None
            if _dbg_leaf:
                # recovered template = the accepted cold-chain source for the
                # traced walker (built before it is folded back in).
                _dbg_tmpl = self._dbg_template(add_coords_in[_dbg_w], _dbg_w)

            # fold the (updated) cold-chain sources back INTO the fit:
            # subtract their templates from the residual (r = d -> d - h_new).
            self.add_back_in_cold_chain_sources(add_coords_in)

            if self.swap_debug:
                _acs_cold_post = asnumpy(self.acs.likelihood())
                _believed = np.asarray(prev_logl[0], dtype=float)
                _delta = _believed - _acs_cold_post
                logger.info(
                    "[STAGE] %s leaf %d POST-READD ACS cold lnL min/med/max "
                    "%.3f / %.3f / %.3f | move-believed cold prev_logl "
                    "min/med/max %.3f / %.3f / %.3f | believed-ACS median "
                    "offset %.3f spread %.3f  (spread ~0 = consistent "
                    "bookkeeping; large spread = state/residual corruption)",
                    self.branch_name, leaf,
                    float(np.min(_acs_cold_post)),
                    float(np.median(_acs_cold_post)),
                    float(np.max(_acs_cold_post)),
                    float(np.min(_believed)),
                    float(np.median(_believed)),
                    float(np.max(_believed)),
                    float(np.median(_delta)),
                    float(np.max(_delta) - np.min(_delta)),
                )

            if os.environ.get("ADDREMOVE_ROUNDTRIP"):
                _rc = np.asarray(removal_coords)[_dbg_w]
                _ac = np.asarray(add_coords)[_dbg_w]
                _same = bool(np.allclose(_rc, _ac))
                _b = self._dbg_snaps.get("before_removal")
                _a = self._dbg_snaps.get("after_readd")
                _dd = lambda x: float(np.sum(np.abs(np.asarray(x)) ** 2)) if x is not None else float("nan")
                # capture the current residual explicitly (after_readd snapshot
                # is only taken under _dbg_leaf below)
                _now = self._dbg_residual(_dbg_w)
                logger.warning(
                    "[ROUNDTRIP %s leaf %d w%d] coords_same=%s  |dcoords|=%.3e  "
                    "sum|resid|^2: before=%.6e  after_readd=%.6e",
                    self.branch_name, leaf, _dbg_w, _same,
                    float(np.max(np.abs(_rc - _ac))), _dd(_b), _dd(_now),
                )

            if _dbg_leaf:
                try:
                    self._dbg_snaps["after_readd"] = self._dbg_residual(_dbg_w)
                    self._debug_plot_source_sequence(
                        leaf, _dbg_w, self._dbg_snaps, _dbg_tmpl,
                        {"domain": getattr(self, "_dbg_domain", "wdm")},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.info("[%s_DEBUG] plot skipped: %r", self._dbg_prefix, exc)
                self._dbg_leaf_done = True

            # read out all betas from temperature controls
            new_state.sub_states[self.branch_name].betas_all[leaf][
                : self.ntemps
            ] = temperature_control_here.betas
            # print(leaf)

            # ll_tmp2 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

            logger.info(f"✓ {self.branch_name} leaf {leaf} complete ({time.time() - tic:.1f}s)")

        # udpate at the end
        logger.info(f"✓ {self.branch_name} proposal complete — all leaves processed ({time.time() - tic:.1f}s total)")
        # new_state.log_like[(temp_inds_update, walker_inds_update)] = logl.flatten()
        # new_state.log_prior[(temp_inds_update, walker_inds_update)] = logp.flatten()
        # print("before computing current likelihood. elapsed: ", time.time() - tic)
        current_ll = (
            self.acs.likelihood()
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        # print("after computing current likelihood. elapsed: ", time.time() - tic)
        if np.any(current_ll < 0.0):
            logger.warning(f"The ACS likelihood should always be positive given the psd contribution, but got {current_ll.min()}")
            logger.warning(f"The minimum proposed likelihood was {prev_logl.min()}.")
            if DEBUG_MODE:
                breakpoint()

        _free_pool()
        # TODO: add check with last used logl

        current_lp = (
            self.priors[self.branch_name]
            .logpdf(new_state.branches[self.branch_name].coords[0, :, :].reshape(-1, ndim))
            .reshape(new_state.branches[self.branch_name].shape[1:-1])
            .sum(axis=-1)
        )

        new_state.log_like[0] = current_ll
        # new_state.log_prior[0] = current_lp
        _free_pool()
        if not hasattr(self, "best_last_ll"):
            self.best_last_ll = current_ll.max()
            self.low_last_ll = current_ll.min()
        # print(self.branch_name, self.best_last_ll, current_ll.max(), current_ll.max() - self.best_last_ll)
        # print(current_ll.max(), self.best_last_ll, current_ll.min(), self.low_last_ll)
        self.best_last_ll = current_ll.max()
        self.low_last_ll = current_ll.min()

        if self.temperature_control is None:
            # this really does not matter
            self.temperature_control = self.temperature_controls[0]

        self.temperature_control.swaps_accepted = self.temperature_controls[0].swaps_accepted

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplimental)
        # breakpoint()
        new_state.log_like[:] = (
            current_ll #self.acs.likelihood()
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

        self.free_gpu_memory()

        # assert np.abs(new_state.log_like[0] - self.acs.get_ll(include_psd_info=True)).max() < 1e-4
        # breakpoint()
        logger.debug(f"mean accepted fraction: {np.mean(self.accepted[0] / self.num_proposals)}. elapsed: {time.time() - tic}")
        return new_state, accepted

    def replace_residuals(self, old_state, new_state):
        """Swap the cold-chain contribution from ``old_state`` to ``new_state``.

        Domain-agnostic: each template is assembled through the containers'
        ``signal_gen`` machinery (see :meth:`_apply_cold_chain_sources`) as a
        :class:`~lisatools.domains.DomainBase` (FD, STFT, WDM, ...) and the
        :class:`AnalysisContainerArray` dispatches the actual residual
        update by basis. The old FD-only implementation hand-assembled
        per-channel arrays from ``self.acs.fd``; that path is now
        unnecessary because :meth:`add_signal_to_residual` /
        :meth:`remove_signal_from_residual` work on any domain.
        """
        for leaf in range(old_state.branches[self.branch_name].shape[-2]):
            removal_coords = old_state.branches[self.branch_name].coords[0, :, leaf]
            self._current_leaf = int(leaf)
            removal_coords_in = self._to_phys(removal_coords)
            self._apply_cold_chain_sources(removal_coords_in, sign=+1)

            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self._to_phys(add_coords)
            self._apply_cold_chain_sources(add_coords_in, sign=-1)

        _free_pool()


# stft_tof addition: multi-GPU variant of the general add/remove move with
# per-device waveform-generator replicas. Kept through the merge; its
# interaction with the dev-side ACA sharding is reviewed in the dedicated
# multi-GPU pass.
class MultiGPUResidualAddRemoveMove(ResidualAddOneRemoveOneMove):
    """Deprecated alias — :class:`ResidualAddOneRemoveOneMove` is ONE
    structure that takes ``dcga=`` directly (2026-07 multi-GPU pass). This
    shim maps the legacy ``MultiGPUResidualAddRemoveMove(dcga, waveform_gen,
    ...)`` signature onto it.
    """

    def __init__(
        self,
        dcga: DomainComputationGroupArray,
        waveform_gen: Any,
        branch_name: str,
        coords_shape: tuple,
        waveform_gen_method: str,
        waveform_gen_kwargs: dict,
        waveform_like_kwargs: dict,
        num_repeats: int,
        transform_fn: TransformContainer,
        priors: ProbDistContainer,
        inner_moves: list,
        Tmax: float = np.inf,
        betas_all: np.ndarray = None,
        permute_every: int = 20,
        pad_out_of_prior: bool = False,
        run_async: bool = False,
        run_threaded: bool = False,
        waveform_like_method: str = None,
        **kwargs,
    ):
        import warnings

        warnings.warn(
            "MultiGPUResidualAddRemoveMove is deprecated: construct "
            "ResidualAddOneRemoveOneMove(..., dcga=...) directly (one move "
            "structure).",
            DeprecationWarning,
            stacklevel=2,
        )
        ResidualAddOneRemoveOneMove.__init__(
            self,
            branch_name=branch_name,
            coords_shape=coords_shape,
            waveform_gen=waveform_gen,
            waveform_gen_kwargs=waveform_gen_kwargs,
            waveform_like_kwargs=waveform_like_kwargs,
            acs=dcga.acs,
            num_repeats=num_repeats,
            transform_fn=transform_fn,
            priors=priors,
            inner_moves=inner_moves,
            Tmax=Tmax,
            betas_all=betas_all,
            permute_every=permute_every,
            pad_out_of_prior=pad_out_of_prior,
            dcga=dcga,
            waveform_gen_method=waveform_gen_method,
            waveform_like_method=waveform_like_method,
            run_async=run_async,
            run_threaded=run_threaded,
            **kwargs,
        )
