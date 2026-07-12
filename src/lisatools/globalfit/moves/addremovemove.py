"""Move that swaps individual sources in/out of the residual data array."""

from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from typing import Any, Callable, TYPE_CHECKING

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
from .multigpumove import MultiGPUMoveBase

logger = logging.getLogger(__name__)
DEBUG_MODE = False

if TYPE_CHECKING:
    from ...sources.waveformbase import TDWaveformBase
    from ...domaincomputation import DomainComputationGroupArray
    
class ResidualAddOneRemoveOneMove(GlobalFitMove, StretchMove, Move):
    """
    Move that handles adding and removing sources to and from the residuals stored in the analysis container array.
    This is done by first removing the contribution of the current sources in the cold chain from the residual,
    then proposing new sources for this leaf, and then adding back in the contribution of the new sources to the residual.
    This way we can make sure that the likelihoods are computed correctly for each proposed source and that the likelihoods are consistent with the current state of the residuals in the analysis container array.

    Args:
        branch_name: name of the branch that this move will operate on.
        coords_shape: shape of the coordinates of the sources in the branch that this move will operate on.
        waveform_gen: function that generates the waveforms for the sources given their coordinates.
        waveform_gen_kwargs: keyword arguments for the waveform generator function.
        waveform_like_kwargs: keyword arguments for the likelihood computation function.
        acs: analysis container array that contains the residuals and other information needed for the likelihood computation.
        num_repeats: number of times to repeat the proposal step for each leaf.
        transform_fn: transform container that contains the transforms to be applied to the coordinates before generating waveforms and computing likelihoods.
        priors: prior distribution container that contains the prior distributions for the sources in the branch.
        inner_moves: list of moves and their corresponding weights to be used for proposing new sources for the leaf.
        Tmax: maximum temperature for the temperature control.
        betas_all: array of betas for all leaves and temperatures. Shape is (nleaves_max, ntemps). If None, betas will be initialized as in TemperatureControl.
        permute_every: number of repeats after which to permute the walkers during a temperature swap. This helps with the mixing of the chains.
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
        **kwargs,
    ):

        # Move.__init__(self, **kwargs)
        StretchMove.__init__(self, **kwargs)

        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = coords_shape

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
        coordinate row. Returns ``None`` on any failure (debug must never
        break the sampler).
        """
        try:
            sig = self.waveform_gen(*coords_row_in, **self.waveform_gen_kwargs)
            arr = sig.arr if hasattr(sig, "arr") else sig
            return asnumpy(arr).copy()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s_DEBUG] template build skipped: %r", self._dbg_prefix, exc)
            return None

    @staticmethod
    def _dbg_source_only_ll(residual_arr) -> float:
        """Un-weighted -1/2<r|r> proxy (sum |r|^2) for a residual snapshot title."""
        r = np.asarray(residual_arr)
        return float(-0.5 * np.sum(np.abs(r) ** 2))

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
                a = np.abs(a)
                a = a[None] if a.ndim == 1 else a
                return a[c] if a.shape[0] > c else a[0]

            # One colour scale per channel, shared across template/data and
            # every frame's residual, so the flip-book is calibrated.
            vmax_row = []
            for ch in range(nch):
                pool = [_ch(tmpl, ch), _ch(data, ch)] + [_ch(r, ch) for _, _, r in frames]
                vmax_row.append(max((np.percentile(p, 99.5) for p in pool), default=1.0) or 1.0)

            rr_entry = (
                self._dbg_source_only_ll(snaps["before_removal"])
                if snaps.get("before_removal") is not None else float("nan")
            )
            os.makedirs(self.debug_plot_dir, exist_ok=True)
            base = f"{self.branch_name}_debug_leaf{leaf}_w{walker}_s{self._dbg_step:04d}"

            for fkey, ftitle, residual in frames:
                cols = [("total template", tmpl), ("total data", data),
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
                                slab, aspect="auto", origin="lower", cmap="viridis",
                                vmin=0.0, vmax=vmax_row[ch],
                            )
                            if ci == 2:
                                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                        else:
                            ax.plot(slab.ravel(), lw=0.6)
                            if kind == "fd":
                                ax.set_yscale("log")
                        if ch == 0:
                            ax.set_title(ctitle, fontsize=9)
                        if ci == 0:
                            ax.set_ylabel(chan_labels[ch], fontsize=9)
                rr_here = self._dbg_source_only_ll(residual)
                fig.suptitle(
                    f"{self.branch_name} leaf {leaf} walker {walker} step "
                    f"{self._dbg_step} — {ftitle}  | -0.5<r|r> entry={rr_entry:.3e} "
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

    def _apply_cold_chain_sources(self, coords, sign):
        """One-at-a-time waveform generation + apply to keep peak RAM flat.

        ``sign=-1`` subtracts (add_back), ``sign=+1`` adds (remove). Matches
        the previous batched path semantically but never holds more than a
        single source's waveform in memory.
        """
        import gc
        for i in range(coords.shape[0]):
            sig = self.waveform_gen(*coords[i], **self.waveform_gen_kwargs)
            self.acs.signal_operation(sign, [sig], data_index=np.array([i]))
            del sig
            gc.collect()
            _free_pool()

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
        _free_pool()

        waveforms = []
        for i in range(coords.shape[0]):
            waveforms.append(self.waveform_gen(*coords[i], **self.waveform_gen_kwargs))

        return DomainBaseArray(waveforms)

    def setup_likelihood_here(self, coords):
        """Hook for subclasses to set up state before computing the likelihood."""
        pass

    def compute_acs_like(self, coords_in, data_index, signal_gen, **kwargs):
        """
        Compute the likelihood for the given coordinates and data index using the analysis container array.

        Args:
            coords_in: coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim).
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).
            signal_gen: waveform generator function to use for computing the likelihood. This is needed because in some cases we need to compute the likelihood with a different waveform generator than the one used for proposing new sources, for example when using heterodyned likelihoods.
            kwargs: additional keyword arguments for the likelihood computation function.

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """
        # TODO: we should probably move the prior in here even though
        # in general with current setup it should only be points in the prior
        # that make it here
        # Normalize to a 2D ``(n_sources, ndim)`` batch so the per-source loop
        # below works whether the caller passed a single 1D ``(ndim,)`` source
        # or a 2D batch. ``atleast_2d`` on a 1D vector yields ``(1, ndim)``
        # (row), which is exactly one source -- NOT ndim sources.
        coords_in = get_array_module(coords_in).atleast_2d(coords_in)
        data_index_np = np.atleast_1d(asnumpy(data_index))
        ll = np.full_like(data_index_np, -1e300, dtype=float)
        source_only = kwargs.pop("source_only", False)

        # TODO: VECTORIZATION. Every generator currently wired into this move
        # is SINGLE-SOURCE: EMRI (FEW GenerateEMRIWaveform), SOBBH / MBH
        # (the tdionfly + legacy ResponseWrapper wraps all build ONE combined
        # template per call). The old batched form ``signal_gen(*coords_in.T)``
        # passed each parameter as a length-n_sources VECTOR, which crashes the
        # single-source FEW generator (and would silently sum candidates for
        # the ``combine=True`` tdionfly wraps). So generate per-source in a
        # plain loop for now. To vectorize later, branch on a per-generator
        # capability flag (e.g. ``getattr(signal_gen, "vectorized_over_sources",
        # False)``) and, when True, restore the single batched call — the
        # ``WaveformBase.get_signals_for_residuals`` style already returns a
        # per-source DomainBaseArray that ``all_templates[i]`` can index.
        all_templates = [
            signal_gen(*coords_in[i], **self.waveform_gen_kwargs)
            for i in range(coords_in.shape[0])
        ]

        for i in range(coords_in.shape[0]):
            template = all_templates[i]
            if isinstance(template, np.ndarray) or (
                xp is not np and isinstance(template, xp.ndarray)
            ):
                # raw-array batch output: tag it with the data's settings so
                # template_likelihood receives a DomainBase (DataResidualArray
                # is deprecated -- domains are communicated by settings class).
                settings = self.acs[int(data_index_np[i])].data.settings
                template = settings.associated_class(template, settings)
            ll[i] = self.acs[int(data_index_np[i])].template_likelihood(
                template,
                include_psd_info=not source_only,
                **kwargs,
            )

        return ll

    def compute_like(self, coords_in, data_index):
        """
        Compute the likelihood for the given coordinates and data index.

        Args:
            coords_in: coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim).
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """
        return self.compute_acs_like(
            coords_in, data_index, signal_gen=self.waveform_gen, **self.waveform_like_kwargs
        )

    def setup(self, model, state):
        """Per-iteration setup hook (no-op by default)."""
        return

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
        # shape is (nwalkers, 1 (nleaves_max), ndim)
        ntemps = x[self.branch_name].shape[0]

        coords = x[self.branch_name].reshape(-1, x[self.branch_name].shape[-1])
        data_index_in = np.tile(np.arange(self.nwalkers), (ntemps, 1)).flatten().astype(np.int32)

        coords_in = self.transform_fn.both_transforms(coords)

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

            # remove cold chain sources
            removal_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)
            self.add_back_in_cold_chain_sources(removal_coords_in)

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
            old_coords_in = self.transform_fn.both_transforms(old_coords)

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

            # if hasattr(self, "waveform_gen_method"):
            #     signal_gen = getattr(self.waveform_gen, self.waveform_gen_method)
            # else:
            #     signal_gen = self.waveform_gen
            
            # acs_like_here = self.compute_acs_like(old_coords_in, data_index=data_index_in, signal_gen=signal_gen, source_only=True).reshape((self.ntemps, self.nwalkers)).real
            # diff = prev_logl - acs_like_here

            # if np.any(np.abs(diff) > 1e-1):
            #         logger.warning(f"acs likelihood: {acs_like_here.flatten()}. proposed likelihood: {prev_logl.flatten()}. This could be a sign of numerical issues.")
            #         if DEBUG_MODE:
            #             breakpoint()
            #         else:
            #             raise ValueError(f"Large difference in log likelihood encountered: {np.abs(diff).max()}. This could be a sign of numerical issues.")

            if np.any(prev_logl < -1e10) or np.any(prev_logl > 1e30):
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
            for repeat in tqdm(range(self.num_repeats), desc=f"{self.branch_name} update, leaf {leaf}"):

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
                            new_points_in = self.transform_fn.both_transforms(padded)

                            data_index = np.asarray(walker_inds_here.astype(np.int32))

                            all_logl = self.compute_like(new_points_in, data_index=data_index)
                            logl = np.where(in_prior, all_logl, -1e300)

                        else:
                            new_points_in = self.transform_fn.both_transforms(
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
    
                    if np.any(logl[in_prior] < -1e10) or np.any(logl[in_prior] > 1e30):
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

                fancy_swap = (repeat % self.permute_every == 0) and (repeat > 0)
                #if fancy_swap:
                    # logger.debug(f"Permuting walkers before swap.")
                compute_log_like = self.log_like_for_fancy_swaping

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

                temperature_control_here.adapt_temps()

                new_state.branches_coords[self.branch_name][:, :, leaf] = coords_for_swap[
                    self.branch_name
                ][:, :, 0]

            # ll_tmp1 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

            # add back cold chain sources
            _free_pool()

            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)

            _dbg_tmpl = None
            if _dbg_leaf:
                # recovered template = the accepted cold-chain source for the
                # traced walker (built before it is folded back in).
                _dbg_tmpl = self._dbg_template(add_coords_in[_dbg_w], _dbg_w)

            self.remove_cold_chain_sources(add_coords_in)

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

        Domain-agnostic: the waveform generator returns a
        :class:`~lisatools.domains.DomainBase` (FD, STFT, WDM, ...) and the
        :class:`AnalysisContainerArray` dispatches the actual residual
        update by basis. The old FD-only implementation hand-assembled
        per-channel arrays from ``self.acs.fd``; that path is now
        unnecessary because :meth:`add_signal_to_residual` /
        :meth:`remove_signal_from_residual` work on any domain.
        """
        for leaf in range(old_state.branches[self.branch_name].shape[-2]):
            removal_coords = old_state.branches[self.branch_name].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)
            self._apply_cold_chain_sources(removal_coords_in, sign=+1)

            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)
            self._apply_cold_chain_sources(add_coords_in, sign=-1)

        _free_pool()


# stft_tof addition: multi-GPU variant of the general add/remove move with
# per-device waveform-generator replicas. Kept through the merge; its
# interaction with the dev-side ACA sharding is reviewed in the dedicated
# multi-GPU pass.
class MultiGPUResidualAddRemoveMove(ResidualAddOneRemoveOneMove, MultiGPUMoveBase):
    """
    Wrapper around ResidualAddOneRemoveOneMove that runs the waveform generation and likelihood computation on multiple GPUs.

    Args:
    dcga: DomainComputationGroupArray that contains the information about the domain computation groups and the GPUs to use for each group.
    waveform_gen: waveform generator class that generates the waveforms for the sources given their coordinates.
    branch_name: name of the branch that this move will operate on.
    coords_shape: shape of the coordinates of the sources in the branch that this move will operate on.
    waveform_gen_method: name of the method of the waveform generator class to use for generating the waveforms for residual operations.
    waveform_gen_kwargs: keyword arguments for the waveform generator method.
    waveform_like_kwargs: keyword arguments for the likelihood computation function.
    num_repeats: number of times to repeat the proposal step for each leaf.
    transform_fn: transform container that contains the transforms to be applied to the coordinates before generating waveforms and computing likelihoods.
    priors: prior distribution container that contains the prior distributions for the sources in the branch.
    inner_moves: list of moves and their corresponding weights to be used for proposing new sources for each leaf.
    Tmax: maximum temperature for the temperature control.
    betas_all: array of betas for all leaves and temperatures. Shape is (nleaves_max, ntemps). If None, betas will be initialized as in TemperatureControl.
    permute_every: number of repeats after which to permute the walkers during a temperature swap. 
    pad_out_of_prior: whether to pad proposed sources that are out of the prior bounds to avoid JIT compilation issues. If True, proposed sources that are out of the prior bounds will be replaced with the first in-prior point. 
    run_async: whether to run the waveform generation and likelihood computation asynchronously for each GPU. If True, the synchronization will happen on the python side after the kernel calls. 
    run_threaded: whether to run the waveform generation and likelihood computation in separate threads for each GPU.
    waveform_like_method: name of the method of the waveform generator class to use for generating the waveforms for likelihood computation. If None, will use the same method as waveform_gen_method.
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
        **kwargs
    ):
        ResidualAddOneRemoveOneMove.__init__(
            self,
            branch_name=branch_name,
            coords_shape=coords_shape,
            waveform_gen=getattr(waveform_gen, waveform_gen_method),
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
            **kwargs
        )

        MultiGPUMoveBase.__init__(self, dcga, run_async=run_async, run_threaded=run_threaded)

        self.waveform_gen = waveform_gen
        self.waveform_gen_method = waveform_gen_method
        self.waveform_like_method = waveform_like_method or waveform_gen_method

        self.create_waveform_gen_replicas()

    def create_waveform_gen_replicas(self, ):
        """
        Create replicas of the waveform generator for each GPU.
        """

        self._waveform_generators = []
        for i, device in enumerate(
            self.dcga.gpus if self.dcga.gpus is not None else [None] * self.dcga.num_splits
        ):
            if not hasattr(self.waveform_gen, "kwargs"):
                raise ValueError("Waveform generator must have a 'kwargs' attribute that contains the keyword arguments to initialize the waveform generator.")    
            
            with self.dcga.device_context(device):
                # if i == 0:
                #     # Reuse the initial waveform generator for the first split to save memory
                #     self._waveform_generators.append(self.waveform_gen)
                # else:
                init_kwargs = self.waveform_gen.kwargs.copy()
                if "orbits" in init_kwargs:
                    init_kwargs["orbits"] = self.dcga.computation_groups[i].orbits

                self._waveform_generators.append(
                    self.waveform_gen.__class__(**init_kwargs)
                )

    def free_gpu_memory(self):
        self.dcga.free_gpu_memory()

    @property
    def waveform_generators(self) -> list:
        return self._waveform_generators
    
    def make_args_tuple(self, coords) -> tuple:
        """
        Make a tuple of arguments for the waveform generator from the given coordinates.
        """
        # unpack the coordinates into separate arguments for the waveform generator
        return tuple(xp.array(coords[:, i]) for i in range(coords.shape[1])) # had to do in this way to give JAX the right memory addresses

    def prepare_inputs(self, coords, data_index):
        """
        Prepare the inputs for the waveform generator from the given coordinates and data index.
        """

        positions_per_split, data_intra_index_per_split, _ = self.dcga.unpack_indices(data_index)
        coords_per_split = self.dcga.unpack_coords(positions_per_split, coords, keep_tuple=True)

        data_intra_index_per_split, coords_per_split = self.dcga.place_on_device(
            items=(data_intra_index_per_split, coords_per_split)
        )

        waveform_args_per_split = self.dcga._loop_operation(
            operation=[self.make_args_tuple for _ in self.dcga.computation_groups],
            operation_args_per_split=coords_per_split,
        )

        return positions_per_split, data_intra_index_per_split, waveform_args_per_split
    
    def aggregate_waveforms(self, waveforms_per_split: list[DomainBaseArray | list[DomainBase]]) -> list[DomainBase]:
        """
        Aggregate the waveforms from each split into a single list of DomainBase objects.
        """
        waveforms = []
        for waveforms_split in waveforms_per_split:
            waveforms.extend(waveforms_split)
        return waveforms

    def get_waveform_here(self, coords: np.ndarray) -> list[DomainBase]:
        """Get the waveforms for the given source coordinates.

        """
        self.free_gpu_memory()

        data_index = np.arange(coords.shape[0], dtype=np.int32)

        _, _, waveform_args_per_split = self.prepare_inputs(coords, data_index)

        operations = [getattr(waveform_gen, self.waveform_gen_method) for waveform_gen in self.waveform_generators]

        waveforms_out = self.dcga._loop_operation(
            operation=operations,
            operation_args_per_split=waveform_args_per_split,
            operation_kwargs=self.waveform_gen_kwargs,
            aggregate_fn=self.aggregate_waveforms,
            run_threaded=self.run_threaded,
        )

        self.dcga.synchronize()

        return waveforms_out

    def setup_likelihood_here(self, coords: np.ndarray) -> None:
        """
        Set up the likelihood computation. In the general case, this means computing the :math:\\langle d | d \\rangle term.
        """

        self.dcga.compute_d_d_terms()

    def compute_like(self, coords_in: np.ndarray, data_index: np.ndarray) -> np.ndarray:
        """
        Compute the likelihood for the given coordinates and data index.

        Args:
            coords_in: coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim).
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).
        
        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """

        positions_per_split, data_intra_index_per_split, waveform_args_per_split = self.prepare_inputs(coords_in, data_index)

        waveform_like_operations = [getattr(waveform_gen, self.waveform_like_method) for waveform_gen in self.waveform_generators]

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

        # Release GPU arrays before freeing the pool.
        # waveform_args_per_split / data_intra_index_per_split are small coord arrays
        # from place_on_device/make_args_tuple; likelihood_args_per_split holds the
        # large template arrays.  All must be dereferenced before free_all_blocks() so
        # those blocks are actually returned to CUDA rather than staying "owned" in the pool.
        del likelihood_args_per_split, waveform_args_per_split, data_intra_index_per_split
        
        self.free_gpu_memory()

        if np.any(~np.isfinite(likelihoods)):
                logger.warning(f"Non-finite likelihoods encountered: {likelihoods}. This could be a sign of numerical issues.")
                if DEBUG_MODE:
                    breakpoint()

        likelihoods = np.where(np.isfinite(likelihoods), likelihoods, -1e300)

        return likelihoods
