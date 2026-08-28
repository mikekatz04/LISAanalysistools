"""MCMC move that updates the LISA noise model (psd / galfor / sgwb branches).

Noise-move split (2026-07): ONE parameterized move class. Each instance
samples ``sampled_branches`` under its OWN temperature ladder while the
likelihood always evaluates the full noise model, with non-sampled noise
branches frozen at their cold-chain rows. ``sampled_branches=None``
reproduces the historical joint psd+galfor+sgwb move exactly.

stft_tof merge (2026-06): hybrid likelihood routing. The move keeps the
dev-side domain-agnostic structure (per-walker sensitivity matrices installed
on the AnalysisContainers, scored via ``acs.likelihood()``), and adds the
stft_tof C++ XYZ-sensitivity kernel (``psd_likelihood_wrap`` shared-memory
sums) as a fast path when the data basis is FD or STFT and the ACA holds a
single shard. The WDM basis falls back to the ACA route until the WDM
counterpart kernels land in domains.cu.
"""

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp
from eryn.model import Model

# Propose-level stage timers, same machinery and same reporting format the GB
# moves use (_ProposeTimer / _tspan in gbspecialstretch). psd + galfor were
# the only uninstrumented branches: with GB at 47.7% of a 1,018 s iteration
# and VGB at 14.4%, the remaining ~38% could only be inferred as a residual.
# gbspecialstretch does not import this module, so there is no cycle.
from .gbspecialstretch import _ProposeTimer, _tspan
from eryn.moves import RedBlueMove, StretchMove
from eryn.state import BranchSupplemental
from eryn.state import State as eryn_State
from eryn.utils.transform import TransformContainer

from tqdm import tqdm

from ...analysiscontainer import AnalysisContainerArray
from ...diagnostic import (
    batched_residual_full_source_and_noise_likelihoods,
    psd_debug_checks_enabled,
)
from ...domaincomputation import DomainComputationGroupArray
from ...domains import (
    CoarseWDMSettings,
    FDSettings,
    FDSignal,
    STFTSettings,
    WDMSettings,
)
from ...sensitivity import (
    _XYZ_ELEMENT_SENS,
    _mat3x3_det_inv,
    CompositeSensitivityBackend,
    GalacticForeground,
    InstrumentNoise,
    SGWB,
    XYZSensitivityBackend,
)
from ...stochastic import check_stochastic
from ...utils.device import device_context
from ...utils.utility import asnumpy
from ..state import GFState
from .globalfitmove import GlobalFitMove

logger = logging.getLogger(__name__)

DEBUG_MODE = False


class PSDMove(GlobalFitMove, StretchMove):
    """
    Noise-model move: samples ``sampled_branches`` of the noise model
    (``psd``, ``galfor``, ``sgwb``) while the likelihood always evaluates the
    FULL noise model — non-sampled noise branches contribute their cold-chain
    coordinates (per walker), the same "temper your own ensemble against the
    cold-chain rest" structure the add/remove source branches use. With the
    default ``sampled_branches=None`` every noise branch present in the state
    is sampled jointly (the historical single-joint-move behavior).

    Args:
        acs: AnalysisContainerArray containing the data and sensitivity information.
        priors: dictionary of priors for the parameters.
        *args: additional arguments for the Move class.
        sampled_branches: which noise branches THIS move samples (subset of
            ``["psd", "galfor", "sgwb"]``). ``None`` samples all present
            branches jointly. Each move instance carries its own
            ``temperature_control`` ladder for its sampled block.
        num_repeats: number of times to repeat the move before returning.
        max_logl_mode: if True, will keep running the move until the maximum log likelihood does not change for a certain number of checks. This is useful for finding the maximum likelihood point.
        psd_kwargs: additional keyword arguments for the psd_log_like function.
        sensitivity_backend: instance of XYZSensitivityBackend to use for computing the likelihood.
        psd_transform_fn: TransformContainer for transforming the PSD parameters.
        galfor_transform_fn: TransformContainer for transforming the galactic
            foreground parameters. NOTE the long-standing order convention is
            preserved as-is: the galfor prior/sampling basis is labelled
            ``[amp, knee, alpha, Slope1, Slope2]`` while the kernel unpack
            reads ``[Amp, alpha, f_1, kn, f_2]`` (see
            :meth:`prepare_likelihood_inputs`; history in the commented
            reorder block in ``run.py``). Do not reorder here.
        sgwb_transform_fn: TransformContainer hook for the SGWB parameters
            (plumbed for symmetry; SGWB coordinates are currently consumed
            untransformed on the ACA route, matching historical behavior).
        permute_every: number of repeats after which to permute the walkers during a temperature swap. This helps with the mixing of the chains.
        tolerance: minimum allowed distance between spline knot positions in the sensitivity model.
        **kwargs: additional keyword arguments for the Move class.

    Env knobs:
        ``PSD_BATCH`` (seeds :attr:`psd_batch`, default ``1``): on the ACA
        (composite / WDM) route, score each walker batch through ONE batched
        covariance build + ONE batched likelihood per shard instead of the
        per-walker Python loop — same numbers (parity asserted at
        rtol <= 1e-12 in ``tests/test_psd_move_batched.py``), O(1) kernel
        launches per scoring batch per device instead of O(nwalkers), and no
        per-batch sens-mat install/restore or ``linear_psd_arr`` repack.
        ``PSD_BATCH=0`` restores the historical per-walker path (the
        ``_force_parent_path``-style escape hatch). The batched route
        engages only when supported (CompositeSensitivityBackend with the
        instrument basis cache, a linear-in-noise-levels instrument model,
        3x3 channel structure, no frequency-layer mask, full-likelihood
        containers); anything else silently keeps the per-walker path.
    """

    # canonical noise-model branch order (psd is mandatory in the model;
    # galfor/sgwb optional)
    NOISE_BRANCHES = ("psd", "galfor", "sgwb")
    # Do not trade runtime for an unbounded second copy of the PSD plane on
    # uncoarsened two-year WDM runs.  The try5 coarse cache is only ~6.4 MiB;
    # the equivalent Q=1 cache would approach 1 GiB for 14 walkers.
    FIXED_COMPONENT_CACHE_MAX_BYTES = 256 * 1024**2
    # Omitted foreground layers perturb every diagonal covariance by less
    # than this fraction of the frozen instrument covariance.  At Q=150 the
    # resulting worst-case accumulated log-likelihood error is O(1e-4), far
    # below both floating Monte Carlo scatter and proposal-scale delta-log-L,
    # while the try5 posterior keeps only ~38 of 59 active layers.
    GALFOR_SUBBAND_REL_TOL = 1e-10
    # A WDM layer averages a finite frequency window.  The cheap layer-center
    # support estimate can cross the relative threshold one layer earlier than
    # that exact average near a sharp foreground cutoff.  Retain two neighbors
    # on either side before performing the exact selected-layer fold.
    GALFOR_SUBBAND_GUARD_LAYERS = 2

    # per-iteration acceptance deltas; allocated by propose() at the module
    # ladder shape, so run_move() skips the tally when called on its own.
    _tally_in_model_proposed = None
    _tally_in_model_accepted = None
    _tally_swaps_proposed = None
    _tally_swaps_accepted = None

    def __init__(
        self,
        acs: AnalysisContainerArray,
        priors,
        *args,
        sampled_branches: list = None,
        num_repeats: int = 1,
        max_logl_mode: bool = False,
        psd_kwargs: dict = {},
        sensitivity_backend: XYZSensitivityBackend = None,
        psd_transform_fn: TransformContainer = None,
        galfor_transform_fn: TransformContainer = None,
        sgwb_transform_fn: TransformContainer = None,
        # 10 (user ruling 2026-08-12): fancy (walker-permuting, re-scored)
        # swap cadence; identity naive swaps fire every repeat regardless.
        permute_every: int = 10,
        tolerance: float = 0.0,
        dcga: DomainComputationGroupArray = None,
        run_async: bool = False,
        run_threaded: bool = False,
        build_threads: int = 1,
        coarse_runtime=None,
        **kwargs,
    ):

        GlobalFitMove.__init__(self, *args, **kwargs)
        StretchMove.__init__(self, *args, **kwargs)
        # ONE move structure (2026-07 multi-GPU pass): pass ``dcga=`` to run
        # the kernel fast path through per-device replicas — how many GPUs
        # the move uses is set by the DCGA it is built with. Without a DCGA
        # the move runs its single-device path. Same object, same propose.
        self._dcga = dcga
        self._run_async = run_async
        self._run_threaded = run_threaded
        # ACA-route per-walker sensitivity builds, spread over threads. See
        # :attr:`build_threads`. Pool is lazy and never pickled.
        self._build_threads = max(1, int(build_threads))
        self._build_pool = None
        # One-time serial warm of the backend's walker-independent caches,
        # tracked PER DEVICE on multi-GPU ACAs (see the note in
        # :meth:`_build_batch`). Holds the devices already warmed (``None``
        # is the CPU / single-device entry).
        self._build_warmed = set()
        # All-source coarse sidecar (plan-2): a CoarseWDMRuntime owning the
        # per-walker coarse statistics and the coarse backend. None (default)
        # leaves every existing route untouched. The runtime is consulted
        # only by compute_coarse_log_like and the propose-entry refresh; the
        # fine callback, the containers' sens_mat, and the packed ACA buffer
        # never see coarse state.
        self.coarse_runtime = coarse_runtime
        self._fixed_component_covariances_coarse = {}
        if dcga is not None:
            if acs is None:
                acs = dcga.acs
            if sensitivity_backend is None:
                sensitivity_backend = (
                    dcga.computation_groups[0].sensitivity_backend
                )
        self.acs = acs
        self.psd_kwargs = psd_kwargs
        self.priors = priors
        self.num_repeats = num_repeats
        self.max_logl_mode = max_logl_mode
        self.starting_now = True

        if sampled_branches is not None:
            unknown = [b for b in sampled_branches if b not in self.NOISE_BRANCHES]
            if unknown:
                raise ValueError(
                    f"Unknown noise branch(es) {unknown}; sampled_branches must be a "
                    f"subset of {list(self.NOISE_BRANCHES)}."
                )
            # canonical order regardless of input order
            sampled_branches = [b for b in self.NOISE_BRANCHES if b in sampled_branches]
        self.sampled_branches = sampled_branches
        # cold-row snapshot of the non-sampled noise branches, refreshed at
        # every propose(); empty when this move samples the whole model
        self._fixed_noise_coords = {}
        # Per-propose, per-walker additive covariances for noise branches that
        # this Metropolis-within-Gibbs move freezes at their cold rows.  These
        # arrays are exact and immutable for the whole repeat block.
        self._fixed_component_covariances = {}
        self._fixed_total_covariances = {}
        self._fixed_total_loglikes = None
        self._fixed_total_loglike_frequency_terms = None

        self.sensitivity_backend = sensitivity_backend
        self.psd_transform_fn = psd_transform_fn
        self.galfor_transform_fn = galfor_transform_fn
        self.sgwb_transform_fn = sgwb_transform_fn

        self.permute_every = permute_every
        self.tolerance = tolerance

        # Debug escape hatch (see the note in the legacy MultiGPUPSDMove):
        # when True, psd_log_like bypasses the DCGA routing entirely.
        self._force_parent_path = False

        # Walker-batched ACA-route scoring (see the class docstring's
        # PSD_BATCH entry). ``_batch_runtime`` caches the per-device
        # walker-independent pieces (instrument bases handles, modulation
        # matrices, extra-component covariances) — built once per device and
        # persisted for the run (allocate-once rule); dropped from pickles.
        self.psd_batch = bool(int(os.environ.get("PSD_BATCH", "1")))
        self._batch_runtime = {}
        self._batch_route_state = None  # lazily resolved: True/False
        self._batch_route_reason = None
        self._batch_mem_logged = set()

        self._setup_debug()

    # Noise-model twin of the per-branch ``{BRANCH}_DEBUG`` plot hooks in
    # addremovemove/gbspecialstretch. The prefix follows the move's sampled
    # branch: ``PSD_DEBUG`` on the psd move, ``GALFOR_DEBUG`` (alias
    # ``FG_DEBUG``) on the galfor move, ``SGWB_DEBUG`` on sgwb. Knobs:
    #   {P}_DEBUG             "1" arms the plot hook (default off)
    #   {P}_DEBUG_DIR         output folder (default ./gf_output/{branch}_debug/)
    #   {P}_DEBUG_PLOT_WALKER walker / analysis container plotted (0)
    #   {P}_DEBUG_EVERY       plot only every Nth propose (1)
    def _setup_debug(self) -> None:
        """Read the ``{PSD,GALFOR/FG,SGWB}_DEBUG`` env options (from __init__)."""
        sampled = list(self.sampled_branches or self.NOISE_BRANCHES)
        if sampled == ["galfor"]:
            prefixes = ("GALFOR", "FG")
        elif sampled == ["sgwb"]:
            prefixes = ("SGWB",)
        else:
            prefixes = ("PSD",)
        self._debug_prefix = prefixes[0]

        def _opt(name, default):
            for p in prefixes:
                v = os.environ.get(f"{p}_{name}")
                if v not in (None, ""):
                    return v
            return default

        self.debug = bool(int(_opt("DEBUG", "0")))
        self.debug_plot_dir = _opt("DEBUG_DIR", f"./gf_output/{sampled[0]}_debug/")
        self.debug_plot_walker = int(_opt("DEBUG_PLOT_WALKER", "0"))
        self.debug_every = max(1, int(_opt("DEBUG_EVERY", "1")))
        self._debug_step = 0
        self._debug_walker_params = None
        if self.debug:
            logger.info(
                "[%s_DEBUG] armed: dir=%s walker=%d every=%d",
                self._debug_prefix, self.debug_plot_dir,
                self.debug_plot_walker, self.debug_every,
            )

    def _debug_plot_psd_vs_residual(self) -> None:
        """Overlay the walker's current noise model on the residual's
        per-layer power. For the WDM basis ``E[w_mn^2] == S_wdm[m]`` (the
        fold validation contract), so on a converged noise fit the two
        curves sit on top of each other; a mismatch localizes which layers
        the fit is missing."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def _h(a):
                return a.get() if hasattr(a, "get") else np.asarray(a)

            w = self.debug_plot_walker
            ac = self.acs[w]
            res = _h(ac.data.arr)
            S = _h(ac.sens_mat[:])

            nch = res.shape[0]
            res_pow = np.abs(res) ** 2
            if res_pow.ndim == 3:
                res_pow = res_pow.mean(axis=-1)
            if S.ndim == res.ndim + 1 and S.shape[0] == S.shape[1]:
                Sd = np.stack([S[i, i] for i in range(nch)])
            else:
                Sd = S[:nch]
            if Sd.ndim == 3:
                Sd = Sd.mean(axis=-1)

            settings = getattr(ac.sens_mat, "basis_settings", None)
            f = _h(settings.f_arr) if hasattr(settings, "f_arr") else None
            n_layer = res_pow.shape[-1]
            if f is None or len(f) != n_layer:
                f = np.arange(n_layer, dtype=float)
            good = f > 0

            # Instrument-only reference (same psd params, foreground/sgwb
            # zeroed): the gap between this and the total model IS the
            # fitted foreground contribution per layer — without it a
            # galfor-absorbed galaxy and no galaxy at all look identical.
            S_instr = None
            dbg = self._debug_walker_params
            if dbg is not None and (dbg[2] is not None or dbg[3] is not None):
                try:
                    Si = _h(self._build_sensitivity_for_walker(
                        dbg[0], dbg[1], None, None)[:])
                    if Si.ndim == res.ndim + 1 and Si.shape[0] == Si.shape[1]:
                        S_instr = np.stack([Si[i, i] for i in range(nch)])
                    else:
                        S_instr = Si[:nch]
                    if S_instr.ndim == 3:
                        S_instr = S_instr.mean(axis=-1)
                except Exception as e:
                    logger.warning(
                        "[%s_DEBUG] instrument-only reference skipped: %r",
                        self._debug_prefix, e,
                    )

            os.makedirs(self.debug_plot_dir, exist_ok=True)
            move = getattr(self, "gf_move_name", self._debug_prefix.lower())
            labels = ("X", "Y", "Z")
            fig, axes = plt.subplots(nch, 1, figsize=(8, 2.6 * nch), sharex=True)
            axes = np.atleast_1d(axes)
            for i in range(nch):
                ch = labels[i] if i < len(labels) else str(i)
                axes[i].loglog(f[good], res_pow[i][good], lw=0.8,
                               label=f"residual E[w^2] ({ch})")
                axes[i].loglog(f[good], np.real(Sd[i])[good], "k--", lw=1.0,
                               label="noise model S (total)")
                if S_instr is not None:
                    axes[i].loglog(f[good], np.real(S_instr[i])[good], "r:",
                                   lw=1.0, label="instrument-only S")
                axes[i].legend(fontsize=7, loc="upper right")
                axes[i].set_ylabel("power")
            axes[-1].set_xlabel("f [Hz]")
            axes[0].set_title(f"{move}: walker {w}, propose {self._debug_step}")
            fname = os.path.join(
                self.debug_plot_dir,
                f"{move}_psd_vs_res_{self._debug_step:05d}.png",
            )
            fig.savefig(fname, dpi=110, bbox_inches="tight")
            plt.close(fig)
            logger.info("[%s_DEBUG] wrote %s", self._debug_prefix, fname)
        except Exception as e:  # debug hooks must never kill a run
            logger.warning(
                "[%s_DEBUG] psd-vs-residual plot skipped: %r",
                self._debug_prefix, e,
            )

    @property
    def dcga(self) -> DomainComputationGroupArray:
        return self._dcga

    @property
    def run_async(self) -> bool:
        return self._run_async

    @property
    def run_threaded(self) -> bool:
        return self._run_threaded

    @property
    def build_threads(self) -> int:
        """Threads used for the ACA route's per-walker sensitivity builds.

        1 (default) keeps the historical serial loop. Higher values spread
        one batch's builds over :attr:`build_pool`; the batching in
        :meth:`compute_log_like` already guarantees one row per walker per
        batch, so the workers touch disjoint ``acs[w]``.

        This pays off only because the work releases the GIL: the LAT
        nanobind kernels are declared with
        ``nb::call_guard<nb::gil_scoped_release>()`` and the composite
        assembly is numpy. Measured on the stock noise grid (0.3-8 mHz =>
        59x1024 active cells): 3.11 ms/walker serial -> 1.73 at 4 threads
        (1.8x) -> 1.78 at 6, i.e. it **saturates at ~4**. The arrays are
        small enough that Python-level orchestration, not the kernels, sets
        the floor, so do not expect linear scaling and do not raise this
        past ~4 hoping for more.

        Output is bitwise-identical to the serial path: the rows are
        independent and nothing is reduced across them.
        """
        return self._build_threads

    @property
    def build_pool(self) -> ThreadPoolExecutor:
        """Lazy pool for :attr:`build_threads` (mirrors ``AnalysisContainerArray``)."""
        if self._build_pool is None:
            self._build_pool = ThreadPoolExecutor(max_workers=self._build_threads)
        return self._build_pool

    @property
    def _skip_linear_psd_repack(self) -> bool:
        """True when the ACA route can leave ``acs.linear_psd_arr`` untouched.

        ``reset_linear_psd_arr`` repacks EVERY container's ``invC`` (a
        ``(3,3,*basis)`` array — 4.5 MB each on the stock noise grid) into the
        ACA's contiguous buffer and rebinds ``sens_mat.invC`` to a view of it.
        The buffer exists for ONE reason: so a single C++/CUDA kernel can take
        one flat pointer spanning all walkers and index it by ``data_index``.
        Its only consumers are ``DomainComputationGroupArray`` (multi-GPU
        replicas) and the GB chunked-heterodyne kernels in ``chunked_het``.

        The ACA likelihood route reads ``ac.sens_mat.invC`` per container and
        never touches the buffer, so on a run with no such kernel the repack is
        pure memcpy — measured at 16.1 ms/call, ~4.7 s of an ~11 s sampling
        iteration on the stock noise fit (~40% of it).

        .. warning::
           **TODO (WDM C++ likelihood kernel).** This skip is only valid while
           the WDM basis has NO batched C++ likelihood kernel — that absence is
           exactly why this move falls back to the ACA route
           (:meth:`_kernel_fast_path_available` accepts FD/STFT only). When the
           WDM kernel lands in ``domains.cu`` it WILL read
           ``linear_psd_arr[0]``, and this must go back to repacking — or the
           kernel will silently score a stale inverse covariance. That failure
           mode is quiet and has bitten before (the walker-0 frozen-readout bug,
           2026-07-10, documented in ``chunked_het.py``). Delete this property
           and its call sites in the same commit that adds the kernel.

           Gating on the backend name is DELIBERATELY conservative but is NOT a
           proof of "no consumer": ``chunked_het`` is backend-dispatched and
           runs on CPU too, so a CPU run that puts a GB chunked-het move and
           this move on ONE ACA would read a stale buffer. That configuration
           does not exist today (the stock CPU noise fits carry no GB branch),
           and the ``_dcga`` guard below covers the multi-GPU consumer.
        """
        if self._dcga is not None:
            return False  # DCGA replicas read the buffer directly
        backend = getattr(self.sensitivity_backend, "backend", None)
        return str(getattr(backend, "name", "")).endswith("_cpu")

    def __getstate__(self):
        """Drop the thread pool from pickles / deepcopies (sprint pickle rule)."""
        state = dict(self.__dict__)
        state["_build_pool"] = None
        state["_fixed_component_covariances"] = {}
        state["_fixed_total_covariances"] = {}
        state["_fixed_total_loglikes"] = None
        state["_fixed_total_loglike_frequency_terms"] = None
        # runtime device arrays (per-device bases/modulations/extras);
        # purely derived state, rebuilt on first use in the copy
        state["_batch_runtime"] = {}
        return state

    # ------------------------------------------------------------------
    # sampled-vs-fixed noise-branch resolution
    # ------------------------------------------------------------------

    def _noise_model_branches(self, state) -> list:
        """Noise branches present in ``state`` (the full noise MODEL)."""
        return [key for key in self.NOISE_BRANCHES if key in state.branches_coords]

    def _resolve_sampled(self, state) -> list:
        """Branches THIS move samples; validated against the state's model."""
        model = self._noise_model_branches(state)
        if self.sampled_branches is None:
            return model
        missing = [b for b in self.sampled_branches if b not in model]
        if missing:
            raise ValueError(
                f"PSDMove configured to sample {self.sampled_branches} but branch(es) "
                f"{missing} are not in the state (noise model: {model})."
            )
        return list(self.sampled_branches)

    def _merged_noise_rows(self, coords, logp_keep, walker_inds_keep) -> dict:
        """Per-row full-noise-model coordinates (sampling basis).

        Sampled branches come from the proposal ``coords`` dict (rows
        surviving the prior cut); fixed branches come from the cold-row
        snapshot taken at ``propose`` entry, indexed by each row's physical
        walker. This is the single seam both likelihood routes consume.
        """
        merged = {}
        for key in self.NOISE_BRANCHES:
            if key in coords:
                merged[key] = coords[key][logp_keep][:, 0]
            elif key in self._fixed_noise_coords:
                merged[key] = self._fixed_noise_coords[key][walker_inds_keep]
        return merged

    def _cold_noise_log_prior(self, state) -> np.ndarray:
        """Sum of ALL noise-branch priors at the cold-row coordinates.

        Written into the engine ``log_prior[0]`` so the cold prior does not
        flip-flop between per-branch partial priors as split noise moves run
        in sequence within a stage.
        """
        lp = None
        for key in self._noise_model_branches(state):
            coords_b = np.asarray(state.branches_coords[key][0])
            ndim = coords_b.shape[-1]
            vals = (
                self.priors[key]
                .logpdf(coords_b.reshape(-1, ndim))
                .reshape(coords_b.shape[:2])
                .sum(axis=-1)
            )
            lp = vals if lp is None else lp + vals
        return lp

    # ------------------------------------------------------------------
    # stft_tof kernel fast path
    # ------------------------------------------------------------------

    def transform_coords(self, coords: list, return_cupy: bool = False, xp=None):
        """
        Prepare the coordinates for the move. This can include transforming the parameters if necessary.

        Args:
            coords: list of numpy arrays containing the parameters for the move. The first element should be the PSD parameters, and the second element (if present) should be the galactic foreground parameters.
            return_cupy: if True, will return the transformed coordinates as cupy arrays for use on the GPU. If False, will return numpy arrays.
            xp: the array library to use (numpy or cupy).
        Returns:
            A tuple containing the transformed PSD parameters and galactic foreground parameters (if present) in the target array library.
        """
        if xp is None:
            xp = self.acs.xp  # Use the array library from the analysis container

        if self.psd_transform_fn is not None:
            psd_pars = self.psd_transform_fn.both_transforms(coords[0])
        else:
            psd_pars = coords[0]

        if len(coords) == 1:
            galfor_pars = np.tile(np.array([1e-200, 1e-3, 1.0, 1.0, 1.0]), (psd_pars.shape[0], 1))
        else:
            if self.galfor_transform_fn is not None:
                galfor_pars = self.galfor_transform_fn.both_transforms(coords[1])
            else:
                galfor_pars = coords[1]

        if return_cupy:
            psd_pars = xp.asarray(psd_pars)
            galfor_pars = xp.asarray(galfor_pars)

        return psd_pars, galfor_pars

    def prepare_likelihood_inputs(self, psd_pars, galfor_pars) -> tuple:
        """
        Prepare the inputs for the likelihood computation. This can include placing the data on the correct device and formatting the parameters as needed.
        """

        xp = self.acs.xp  # Use the appropriate array library (numpy or cupy)

        # Use .copy() (not xp.ascontiguousarray) to guarantee each kernel input
        # owns an independent allocation. ascontiguousarray is a no-op when the
        # slice is already contiguous — which is the case for a column of a
        # Fortran-order 2D array. In that case Soms_d_in_all and Sa_a_in_all
        # would alias psd_pars's buffer, and the CUDA kernel would read through
        # those aliased pointers — exposing it to cupy memory-pool reuse
        # hazards while the kernel runs asynchronously. .copy() forces a
        # fresh, contiguous, owning allocation.
        Soms_d_in_all = psd_pars[:, 0].copy()
        Sa_a_in_all = psd_pars[:, 1].copy()

        if self.sensitivity_backend.use_splines:
            knots_positions = xp.asarray(psd_pars[:, 3::2])
            knots_amplitudes = xp.asarray(psd_pars[:, 2:-1:2])
            half = knots_positions.shape[1] // 2  # Get the mid of the array
            # put the 2 noise levels on the batch axis
            spline_knots_amplitude = xp.stack(
                (knots_amplitudes[:, :half], knots_amplitudes[:, half:])
            )
            spline_knots_position = xp.stack(
                (knots_positions[:, :half], knots_positions[:, half:])
            )

            # Sort
            sort_indices = xp.argsort(spline_knots_position, axis=2)
            # Apply the same indices to both arrays
            spline_knots_position = xp.take_along_axis(
                spline_knots_position, sort_indices, axis=2
            )
            spline_knots_amplitude = xp.take_along_axis(
                spline_knots_amplitude, sort_indices, axis=2
            )
            # now check if any knot position is not too close together
        else:
            spline_knots_position = None
            spline_knots_amplitude = None

        # Same defensive copy for the galactic-foreground components. The
        # previous code unpacked `*xp.ascontiguousarray(galfor_pars.T)` which,
        # for an already-contiguous input, would unpack 5 views into the same
        # base array — same aliasing hazard as Soms/Sa. Copying each row
        # independently gives 5 owning 1D arrays.
        galfor_pars_T = galfor_pars.T  # view; transpose is always free
        Amp_all = galfor_pars_T[0].copy()
        alpha_all = galfor_pars_T[1].copy()
        f_1_all = galfor_pars_T[2].copy()
        kn_all = galfor_pars_T[3].copy()
        f_2_all = galfor_pars_T[4].copy()

        likelihood_args = (
            Soms_d_in_all,
            Sa_a_in_all,
            Amp_all,
            alpha_all,
            f_1_all,
            kn_all,
            f_2_all,
            spline_knots_position,
            spline_knots_amplitude,
        )

        return likelihood_args

    def psd_log_like(self, x: list, supps=None, **sens_kwargs) -> np.ndarray:
        """
        Internal method to compute the log likelihood for the PSD parameters via the C++ XYZ sensitivity kernel (stft_tof fast path).

        Args:
            x: list of numpy arrays containing the parameters for the move. The first element should be the PSD parameters, and the second element (if present) should be the galactic foreground parameters.
            supps: supplemental information for the likelihood computation, such as walker indices.
            **sens_kwargs: additional keyword arguments to pass to the sensitivity backend for likelihood computation.
        Returns:
            A numpy array containing the log likelihood values for each set of parameters.
        """
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        if self._dcga is not None and not getattr(self, "_force_parent_path", False):
            # GPU-spread path: per-device replicas via the DCGA this move was
            # constructed with (one move structure; see __init__).
            return self._psd_log_like_dcga(x, supps=supps, **sens_kwargs)

        xp = self.acs.xp  # Use the appropriate array library (numpy or cupy)

        wi = supps["walker_inds"]
        data_index_all = xp.asarray(wi).astype(np.int32)

        psd_pars, galfor_pars = self.transform_coords(x, return_cupy=True, xp=xp)

        likelihood_args = self.prepare_likelihood_inputs(psd_pars, galfor_pars)

        ll = self.sensitivity_backend.compute_log_like(
            self.acs.linear_data_arr[0], data_index_all, *likelihood_args
        )

        if likelihood_args[-2] is not None and likelihood_args[-1] is not None:
            invalid_knots = xp.any(
                xp.diff(10 ** likelihood_args[-2], axis=2) < self.tolerance, axis=(0, 2)
            )
        else:
            invalid_knots = xp.zeros(psd_pars.shape[0], dtype=bool)

        ll[invalid_knots] = -1e300

        return ll.get() if hasattr(ll, "get") else ll

    def _kernel_fast_path_available(self, has_sgwb: bool = False) -> bool:
        """True when the C++ XYZ sensitivity kernel can score this proposal.

        Conditions: a sensitivity backend is configured, the data basis is FD
        or STFT (the kernel's two domains — the WDM counterpart is planned in
        domains.cu), the ACA holds a single shard (the kernel reads
        ``linear_data_arr[0]`` directly), and the noise MODEL carries no sgwb
        branch — the kernel has no SGWB component, so scoring an sgwb-bearing
        model through it (sampled or fixed) would silently drop that term.
        """
        if has_sgwb:
            return False
        if self.sensitivity_backend is None:
            return False
        basis = self.sensitivity_backend.basis_settings
        if not isinstance(basis, (FDSettings, STFTSettings)):
            return False
        # A DCGA-configured move scores each shard on its own replica, so
        # the kernel path stays available on multi-shard ACAs.
        return self._dcga is not None or len(self.acs.linear_data_arr) == 1

    # ------------------------------------------------------------------
    # dev ACA path + hybrid dispatch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # multi-GPU walker-shard plumbing (ACA route)
    # ------------------------------------------------------------------
    # The ACA shards walkers across devices (``acs.gpu_map`` /
    # ``acs.gpu_splits``); scoring already runs per shard inside the owning
    # device context (``AnalysisContainerArray._vectorized_dispatch``). The
    # helpers below give the per-walker sensitivity BUILDS the same
    # discipline: each build enters the owning walker's ``device_context``
    # and evaluates against a per-device domain-settings replica (the
    # ``source_runtime`` precedent), so nothing reads the primary device's
    # caches cross-device. CPU / single-GPU paths are byte-identical to the
    # historical behavior (device ``None`` -> nullcontext + shared objects).

    def _walker_device(self, walker_index: int):
        """The CUDA device owning walker ``walker_index``'s shard (None on CPU)."""
        # getattr chain: some unit tests build the move via ``__new__`` with
        # no ACA at all — treat that like the CPU path.
        gpus = getattr(getattr(self, "acs", None), "gpus", None)
        if gpus is None:
            return None
        return int(self.acs.gpu_map[int(walker_index)])

    def _backend_settings_for_device(self, device):
        """Per-device replica of the sensitivity backend's domain settings.

        CPU / single-GPU / the run's primary device return the backend's own
        (shared) ``basis_settings`` unchanged — zero extra memory, identical
        objects. A non-primary device returns the cached device-local replica
        from ``source_runtime._device_local_domain_settings_on`` (built once
        per (settings, device) and kept for the run), whose WDM window /
        fold caches live on that device.
        """
        backend = self.sensitivity_backend
        settings = getattr(backend, "basis_settings", None)
        gpus = getattr(getattr(self, "acs", None), "gpus", None)
        if settings is None or device is None or gpus is None:
            return settings
        primary = int(gpus[0])
        if int(device) == primary:
            return settings
        from ..stock.erebor.source_runtime import (
            _device_local_domain_settings_on,
        )

        return _device_local_domain_settings_on(
            settings, self.acs.xp, int(device), primary
        )

    @staticmethod
    def _to_physical(transform_fn, params):
        """One branch's sampling-basis row -> the physical basis the model wants.

        ``both_transforms`` can hand back a leading axis for a single row, so
        squeeze it the way :meth:`SensitivityBackendBase.__call__` does.
        """
        if transform_fn is None or params is None:
            return params
        out = transform_fn.both_transforms(np.asarray(params, dtype=float))
        return np.atleast_1d(np.asarray(out).squeeze())

    def _build_sensitivity_for_walker(
        self, walker_index: int, psd_params, galfor_params, sgwb_params=None
    ):
        """Build the per-walker sensitivity matrix for the given parameters.

        Takes every branch's row in ITS SAMPLING BASIS and applies that
        branch's transform here, so a transformed branch behaves the same on
        this route as on the kernel fast path (:meth:`transform_coords`) --
        the backend only ever knows about the psd transform, so a log-sampled
        galfor/sgwb would otherwise reach the model unexponentiated.

        Routes through the configured :class:`XYZSensitivityBackend`. The
        returned :class:`SensitivityMatrix` is what will be installed on the
        AnalysisContainer for the matching walker when we accept proposals
        (see :meth:`propose`).

        Multi-GPU: the build runs inside the walker's owning
        ``device_context`` and (composite backend) against a per-device
        domain-settings replica, so the produced matrix — covariance,
        ``invC``, ``detC`` — is resident on the walker's shard device.
        """
        # ``sgwb_params`` is only forwarded when present so the legacy
        # XYZSensitivityBackend (whose __call__ has no sgwb kwarg) keeps
        # working for runs without an sgwb branch.
        psd_params = self._to_physical(self.psd_transform_fn, psd_params)
        galfor_params = self._to_physical(self.galfor_transform_fn, galfor_params)
        sgwb_params = self._to_physical(self.sgwb_transform_fn, sgwb_params)
        fixed = getattr(self, "_fixed_component_covariances", {}).get(
            int(walker_index), {}
        )
        if fixed and hasattr(self.sensitivity_backend, "matrix_from_params"):
            return self.sensitivity_backend.matrix_from_params(
                f"walker_{walker_index}",
                None if "psd" in fixed else psd_params,
                galfor_params=None if "galfor" in fixed else galfor_params,
                sgwb_params=None if "sgwb" in fixed else sgwb_params,
                fixed_covariances=fixed,
            )

        extra = {} if sgwb_params is None else dict(sgwb_params=sgwb_params)
        dev = self._walker_device(walker_index)
        settings_here = self._backend_settings_for_device(dev)
        if settings_here is not None and settings_here is not getattr(
            self.sensitivity_backend, "basis_settings", None
        ):
            # only forwarded when it differs so duck-typed / legacy backends
            # without the kwarg keep working on the shared-settings path
            extra["basis_settings"] = settings_here
        xp = getattr(getattr(self, "acs", None), "xp", np)
        with device_context(xp, dev):
            # NB: psd_params / galfor_params were already mapped to physical
            # units at the top of this method (needed by the frozen-component
            # route above), so they must NOT be transformed a second time here.
            return self.sensitivity_backend(
                f"walker_{walker_index}",
                psd_params,
                galfor_params=galfor_params,
                **extra,
            )

    def _fixed_component_cache_available(self) -> bool:
        """Whether exact additive-component caching is safe on this route."""
        if self._dcga is not None:
            return False
        backend = getattr(self.sensitivity_backend, "backend", None)
        return (
            str(getattr(backend, "name", "")).endswith("_cpu")
            and hasattr(self.sensitivity_backend, "component_covariance")
        )

    def _prepare_fixed_component_covariances(self) -> None:
        """Build each frozen branch once per walker for this proposal block."""
        self._fixed_component_covariances = {}
        self._fixed_total_covariances = {}
        self._fixed_total_loglikes = None
        self._fixed_total_loglike_frequency_terms = None
        if not self._fixed_noise_coords or not self._fixed_component_cache_available():
            return

        transforms = {
            "psd": self.psd_transform_fn,
            "galfor": self.galfor_transform_fn,
            "sgwb": self.sgwb_transform_fn,
        }
        tasks = [
            (branch, w, rows[w])
            for branch, rows in self._fixed_noise_coords.items()
            for w in range(len(rows))
        ]

        def _one(task):
            branch, w, params = task
            physical = self._to_physical(transforms[branch], params)
            covariance = self.sensitivity_backend.component_covariance(
                branch, physical, name=f"fixed_{branch}_{w}"
            )
            return branch, int(w), covariance

        # Warm shared basis/spectral caches without a race, then spread the
        # remaining independent rows exactly like the ordinary build path.
        results = []
        task_count = len(tasks)
        if tasks:
            first = _one(tasks.pop(0))
            estimated_bytes = int(getattr(first[2], "nbytes", 0)) * task_count
            if estimated_bytes > self.FIXED_COMPONENT_CACHE_MAX_BYTES:
                logger.info(
                    "skipping frozen noise-component cache: estimated %.1f MiB "
                    "exceeds %.1f MiB limit",
                    estimated_bytes / 1024**2,
                    self.FIXED_COMPONENT_CACHE_MAX_BYTES / 1024**2,
                )
                return
            results.append(first)
            # per-device warm bookkeeping (see :meth:`_build_batch`); this
            # route is CPU-only, so the one task warms the whole set.
            self._build_warmed.add(self._walker_device(first[1]))
        if self._build_threads > 1 and len(tasks) > 1:
            results.extend(self.build_pool.map(_one, tasks))
        else:
            results.extend(_one(task) for task in tasks)
        for branch, w, covariance in results:
            self._fixed_component_covariances.setdefault(w, {})[branch] = covariance

        # A galfor-only split move changes no other model component.  Keep the
        # exact full-band frozen sum so its likelihood can serve as a baseline;
        # each proposal then recomputes only the layers where foreground power
        # measurably changes that sum.  Fixed extras are not represented in the
        # branch cache, so retain the ordinary full-band path when present.
        if (
            self.sampled_branches == ["galfor"]
            and not getattr(self.sensitivity_backend, "extra_components", ())
        ):
            required = set(self._fixed_noise_coords)
            for w, components in self._fixed_component_covariances.items():
                if set(components) != required:
                    self._fixed_total_covariances = {}
                    break
                ordered = [
                    components[key]
                    for key in self.NOISE_BRANCHES
                    if key in components
                ]
                total = ordered[0]
                for covariance in ordered[1:]:
                    total = total + covariance
                self._fixed_total_covariances[w] = total

    def _coarse_batch_fast_path_available(self) -> bool:
        """True for the exact CPU coarse-WDM batched reduction."""
        if not self._fixed_component_cache_available():
            return False
        if not hasattr(self.sensitivity_backend, "covariance_from_params"):
            return False
        acs = self.acs.flatten()
        if len(acs) == 0 or getattr(acs[0], "coarse_stats", None) is None:
            return False
        stat = acs[0].coarse_stats
        return all(getattr(ac, "coarse_stats", None) is stat for ac in acs)

    def _galfor_subband_fast_path_available(self) -> bool:
        """Whether a galfor-only move can score against a frozen baseline."""
        return (
            self._coarse_batch_fast_path_available()
            and self.sampled_branches == ["galfor"]
            and len(self._fixed_total_covariances) == len(self.acs.flatten())
            and hasattr(self.sensitivity_backend, "galfor_coarse_profile")
            and hasattr(
                self.sensitivity_backend,
                "galfor_coarse_profile_estimate",
            )
            and hasattr(
                self.sensitivity_backend,
                "galfor_coarse_covariance_from_profile",
            )
        )

    def _galfor_profile_frequency_mask(self, profile, fixed_covariance):
        """Layers whose foreground diagonal is non-negligible vs ``fixed``."""
        column, modulation = profile
        column = np.asarray(column)
        modulation = np.asarray(modulation)
        if modulation.ndim == 2:
            modulation_diagonal = np.diag(modulation)[:, None]
        else:
            modulation_diagonal = np.stack(
                [modulation[channel, channel] for channel in range(3)], axis=0
            )
        fixed_diagonal = np.real(
            np.stack(
                [fixed_covariance[channel, channel] for channel in range(3)],
                axis=0,
            )
        )
        numerator = (
            np.abs(modulation_diagonal)[:, None, :]
            * np.abs(column)[None, :, None]
        )
        denominator = np.abs(fixed_diagonal)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                denominator > 0.0,
                numerator / denominator,
                np.where(numerator > 0.0, np.inf, 0.0),
            )
        return np.any(
            ratio > self.GALFOR_SUBBAND_REL_TOL, axis=(0, 2)
        )

    def _compute_galfor_subband_loglike(
        self, stat, walker_inds_keep, galfor_coords
    ):
        """Full baseline plus band-limited foreground likelihood corrections."""
        from ...coarsewdm import (
            coarse_wdm_log_likelihood_batch,
            coarse_wdm_log_likelihood_batch_frequency_terms,
        )

        nwalkers = len(self.acs.flatten())
        if self._fixed_total_loglikes is None:
            fixed_batch = np.stack(
                [self._fixed_total_covariances[w] for w in range(nwalkers)],
                axis=0,
            )
            self._fixed_total_loglike_frequency_terms = np.asarray(
                coarse_wdm_log_likelihood_batch_frequency_terms(
                    stat, fixed_batch
                )
            )
            self._fixed_total_loglikes = np.sum(
                self._fixed_total_loglike_frequency_terms, axis=1
            )

        out = np.empty(walker_inds_keep.shape[0], dtype=float)
        remaining = np.arange(walker_inds_keep.shape[0])
        while remaining.size:
            _, first = np.unique(walker_inds_keep[remaining], return_index=True)
            batch = remaining[np.sort(first)]
            physical_params = []
            masks = []
            for row in batch:
                row = int(row)
                w = int(walker_inds_keep[row])
                physical = self._to_physical(
                    self.galfor_transform_fn, galfor_coords[row]
                )
                estimate = (
                    self.sensitivity_backend.galfor_coarse_profile_estimate(
                        physical
                    )
                )
                if estimate is None:
                    return None
                physical_params.append(physical)
                masks.append(
                    self._galfor_profile_frequency_mask(
                        estimate, self._fixed_total_covariances[w]
                    )
                )

            support = np.any(masks, axis=0)
            guard = self.GALFOR_SUBBAND_GUARD_LAYERS
            if guard and np.any(support):
                expanded = support.copy()
                for shift in range(1, guard + 1):
                    expanded[shift:] |= support[:-shift]
                    expanded[:-shift] |= support[shift:]
                support = expanded
            frequency_indices = np.flatnonzero(support)
            walkers = np.asarray(walker_inds_keep[batch], dtype=int)
            if frequency_indices.size == 0:
                out[batch] = self._fixed_total_loglikes[walkers]
            else:
                profiles = [
                    self.sensitivity_backend.galfor_coarse_profile(
                        physical,
                        frequency_indices=frequency_indices,
                    )
                    for physical in physical_params
                ]
                if any(profile is None for profile in profiles):
                    return None
                fixed_subband = np.stack(
                    [
                        self._fixed_total_covariances[w][
                            :, :, frequency_indices, :
                        ]
                        for w in walkers
                    ],
                    axis=0,
                )
                foreground = np.stack(
                    [
                        self.sensitivity_backend.galfor_coarse_covariance_from_profile(
                            profile, frequency_indices
                        )
                        for profile in profiles
                    ],
                    axis=0,
                )
                fixed_subband_logl = np.sum(
                    self._fixed_total_loglike_frequency_terms[
                        walkers[:, None], frequency_indices[None, :]
                    ],
                    axis=1,
                )
                total_subband_logl = coarse_wdm_log_likelihood_batch(
                    stat,
                    fixed_subband + foreground,
                    frequency_indices=frequency_indices,
                )
                out[batch] = (
                    self._fixed_total_loglikes[walkers]
                    + np.asarray(total_subband_logl)
                    - np.asarray(fixed_subband_logl)
                )
            remaining = np.setdiff1d(remaining, batch)
        return out

    def _build_covariance_batch(
        self, batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
    ):
        """Build raw covariances for one-walker-per-row coarse batch."""
        def _one(row):
            row = int(row)
            w = int(walker_inds_keep[row])
            psd_here = self._to_physical(self.psd_transform_fn, psd_coords[row])
            galfor_here = (
                None
                if galfor_coords is None
                else self._to_physical(self.galfor_transform_fn, galfor_coords[row])
            )
            sgwb_here = (
                None
                if sgwb_coords is None
                else self._to_physical(self.sgwb_transform_fn, sgwb_coords[row])
            )
            fixed = self._fixed_component_covariances.get(w, {})
            return self.sensitivity_backend.covariance_from_params(
                f"walker_{w}",
                None if "psd" in fixed else psd_here,
                galfor_params=None if "galfor" in fixed else galfor_here,
                sgwb_params=None if "sgwb" in fixed else sgwb_here,
                fixed_covariances=fixed,
            )

        rows = list(batch)
        covariances = []
        if rows:
            # serial per-device cache warm (see :meth:`_build_batch`); this
            # route is CPU-only, so the first row warms every later one.
            dev = self._walker_device(int(walker_inds_keep[int(rows[0])]))
            if dev not in self._build_warmed:
                self._build_warmed.add(dev)
                covariances.append(_one(rows.pop(0)))
        if self._build_threads > 1 and len(rows) > 1:
            covariances.extend(self.build_pool.map(_one, rows))
        else:
            covariances.extend(_one(row) for row in rows)
        return np.stack(covariances, axis=0)

    # ------------------------------------------------------------------
    # All-source coarse sidecar (plan-2): surrogate scoring seam
    # ------------------------------------------------------------------
    @property
    def coarse_sidecar_active(self) -> bool:
        """True when this move carries a live sidecar runtime (mode != off)."""
        rt = self.coarse_runtime
        return rt is not None and getattr(rt, "mode", "off") != "off"

    def _resolve_mode_like_fns(self):
        """``(mode, init_fn, model_fn, swap_fn)`` for the current coarse mode.

        ``init_fn`` seeds the working state's log-likes (the state invariant),
        ``model_fn`` scores eryn's stage-1 proposals, ``swap_fn`` feeds the
        temperature swaps. off: fine everywhere. search_approx: coarse
        everywhere (an optimizer of the surrogate, not a posterior sampler).
        delayed_acceptance: FINE state invariant + coarse stage-1 + fine
        swaps — stage 2 in :meth:`_propose_delayed_acceptance` closes the
        kernel on the fine target.
        """
        mode = self.coarse_runtime.mode if self.coarse_sidecar_active else "off"
        if mode == "search_approx":
            return (
                mode,
                self.compute_coarse_log_like,
                self.compute_coarse_log_like,
                self.compute_coarse_log_like,
            )
        if mode == "delayed_acceptance":
            return (
                mode,
                self.compute_log_like,
                self.compute_coarse_log_like,
                self.compute_log_like,
            )
        return mode, self.compute_log_like, self.compute_log_like, self.compute_log_like

    def _prepare_fixed_component_covariances_coarse(self) -> None:
        """Coarse twin of :meth:`_prepare_fixed_component_covariances`.

        Frozen-branch covariances on the COARSE grid for the sidecar's
        candidate assembly. Rebuilt every proposal block (the cold rows a
        preceding noise move accepted may have moved); coarse arrays are ~Q
        times smaller than fine, so this is cheap and stays serial.
        """
        self._fixed_component_covariances_coarse = {}
        if not self.coarse_sidecar_active or not self._fixed_noise_coords:
            return
        backend = self.coarse_runtime.coarse_backend
        transforms = {
            "psd": self.psd_transform_fn,
            "galfor": self.galfor_transform_fn,
            "sgwb": self.sgwb_transform_fn,
        }
        for branch, rows in self._fixed_noise_coords.items():
            for w in range(len(rows)):
                physical = self._to_physical(transforms[branch], rows[w])
                covariance = backend.component_covariance(
                    branch, physical, name=f"coarse_fixed_{branch}_{w}"
                )
                self._fixed_component_covariances_coarse.setdefault(int(w), {})[
                    branch
                ] = covariance

    def _build_coarse_covariance_batch(
        self, batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
    ):
        """Coarse candidate covariances for one one-row-per-walker batch."""
        runtime = self.coarse_runtime
        backend = runtime.coarse_backend

        def _one(row):
            row = int(row)
            w = int(walker_inds_keep[row])
            psd_here = self._to_physical(self.psd_transform_fn, psd_coords[row])
            galfor_here = (
                None
                if galfor_coords is None
                else self._to_physical(self.galfor_transform_fn, galfor_coords[row])
            )
            sgwb_here = (
                None
                if sgwb_coords is None
                else self._to_physical(self.sgwb_transform_fn, sgwb_coords[row])
            )
            fixed = self._fixed_component_covariances_coarse.get(w, {})
            return backend.covariance_from_params(
                f"coarse_walker_{w}",
                None if "psd" in fixed else psd_here,
                galfor_params=None if "galfor" in fixed else galfor_here,
                sgwb_params=None if "sgwb" in fixed else sgwb_here,
                fixed_covariances=fixed,
            )

        rows = [_one(row) for row in batch]
        return runtime.xp.stack([runtime.xp.asarray(r) for r in rows], axis=0)

    def compute_coarse_log_like(
        self, coords, inds=None, logp=None, supps=None, branch_supps=None
    ):
        """Surrogate PSD/GALFOR log-likelihood against the coarse sidecar.

        Same eryn row semantics as :meth:`compute_log_like` (prior mask,
        ``walker_inds`` row mapping, frozen-branch merge), scored against the
        sidecar runtime's per-walker coarse statistics. Never touches
        ``state.log_like``, any container's ``sens_mat``, or the packed ACA
        buffer — the values live only in move-local arrays.
        """
        if not self.coarse_sidecar_active:
            raise RuntimeError(
                "compute_coarse_log_like needs a live coarse sidecar runtime "
                "(coarse_runtime with mode != 'off')."
            )
        if logp is None:
            logp = self.compute_log_prior(
                coords, inds=inds, supps=supps, branch_supps=branch_supps
            )
        logl = np.full_like(logp, -1e300)
        logp_keep = ~np.isinf(logp)
        if not np.any(logp_keep):
            return logl, None
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        walker_inds_all = np.asarray(supps.holder["walker_inds"]).reshape(logp.shape)
        walker_inds_keep = walker_inds_all[logp_keep].astype(int)
        merged = self._merged_noise_rows(coords, logp_keep, walker_inds_keep)
        psd_coords = merged["psd"]
        galfor_coords = merged.get("galfor")
        sgwb_coords = merged.get("sgwb")

        tmp_logl = np.empty(walker_inds_keep.shape[0], dtype=float)
        remaining = np.arange(walker_inds_keep.shape[0])
        while remaining.size:
            _, first = np.unique(walker_inds_keep[remaining], return_index=True)
            batch = remaining[np.sort(first)]
            covariances = self._build_coarse_covariance_batch(
                batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
            )
            tmp_logl[batch] = np.asarray(
                asnumpy(
                    self.coarse_runtime.coarse_log_like_batch(
                        covariances, walker_inds_keep[batch]
                    )
                ),
                dtype=float,
            )
            remaining = np.setdiff1d(remaining, batch)
        logl[logp_keep] = tmp_logl
        return logl, None

    def _build_batch(self, batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords):
        """Install the proposed sensitivity matrix for every row in ``batch``.

        ``batch`` holds at most one row per walker (see the batching in
        :meth:`compute_log_like`), so each row owns its ``acs[w]`` outright
        and the rows are independent. With :attr:`build_threads` > 1 they
        run concurrently in :attr:`build_pool`.

        The first row seen for EACH device is always built serially: the
        sensitivity backend fills walker-independent caches on its first
        call (``CompositeSensitivityBackend._instrument_basis_cache`` -- the
        per-WDM-time-slice fold under ``--unequal-arm`` is minutes of work),
        those caches are per device (each keyed on the device / its settings
        replica), and letting N threads race into a cold cache would just do
        that work N times. Once a device is warm, every later call on it is
        a lookup.
        """
        def _one(row):
            row = int(row)
            w = int(walker_inds_keep[row])
            galfor_here = None if galfor_coords is None else galfor_coords[row]
            sgwb_here = None if sgwb_coords is None else sgwb_coords[row]
            self.acs[w].sens_mat = self._build_sensitivity_for_walker(
                w, psd_coords[row], galfor_here, sgwb_here
            )

        self._run_rows_per_split(_one, list(batch), walker_inds_keep)

    def _run_rows_per_split(self, one_fn, rows, walker_inds_keep):
        """Run ``one_fn(row)`` for every row, grouped by owning ACA split.

        The first row seen for each not-yet-warm device runs serially first
        (the per-device cache warm described in :meth:`_build_batch`); the
        rest dispatch through ``acs._run_per_split`` (the
        ``addremovemove.compute_acs_like`` / ``_vectorized_dispatch``
        primitive): serial on CPU / single split, one thread per populated
        split under ``run_threaded`` (multi-GPU). ``one_fn`` enters its own
        walker's ``device_context`` (see
        :meth:`_build_sensitivity_for_walker`), so the workers need no extra
        device pinning. Within a split, :attr:`build_threads` > 1 keeps the
        historical intra-split thread spread.

        The move's ``run_threaded`` flag forces threaded dispatch; otherwise
        the ACA's own ``run_threaded`` default decides (the
        ``compute_acs_like`` precedent).
        """
        pending = []
        for row in rows:
            dev = self._walker_device(int(walker_inds_keep[int(row)]))
            if dev not in self._build_warmed:
                # serial per-device cache warm
                self._build_warmed.add(dev)
                one_fn(row)
            else:
                pending.append(row)
        rows = pending
        if not len(rows):
            return

        def _run_group(group_rows):
            if self._build_threads > 1 and len(group_rows) > 1:
                # list() forces consumption so worker exceptions re-raise.
                list(self.build_pool.map(one_fn, group_rows))
            else:
                for row in group_rows:
                    one_fn(row)

        acs = getattr(self, "acs", None)
        if acs is None or not (
            hasattr(acs, "_run_per_split") and hasattr(acs, "_split_rows")
        ):
            _run_group(list(rows))
            return

        rows_arr = np.asarray(list(rows), dtype=int)
        split_to_pos = acs._split_rows(
            np.asarray(walker_inds_keep, dtype=int)[rows_arr]
        )

        def _worker(split, positions):
            _run_group(rows_arr[np.asarray(positions, dtype=int)])

        acs._run_per_split(
            _worker, split_to_pos,
            run_threaded=True if self._run_threaded else None,
        )

    # ------------------------------------------------------------------
    # walker-batched ACA-route scoring (PSD_BATCH; tier 3)
    # ------------------------------------------------------------------
    # The per-walker ACA route builds one CompositeSensitivityMatrix per
    # proposal row (2 basis multiplies + a WDM stochastic fold + a 3x3
    # det/inv per walker) and reads one likelihood per container -- O(nb)
    # Python-orchestrated kernel chains per scoring batch. Everything in
    # that chain is either LINEAR in per-walker scalars (the instrument
    # bases), a per-walker 1-D curve feeding a linear fold (the stochastic
    # components), or elementwise over the basis grid (_mat3x3_det_inv, the
    # likelihood sums) -- so a walker axis slots straight in and the whole
    # batch is O(1) kernel launches per device. Parity with the per-walker
    # route is exact by construction (same operations, same order); tested
    # at rtol <= 1e-12 in tests/test_psd_move_batched.py.

    def _batched_route_ready(self) -> bool:
        """Whether the PSD_BATCH walker-batched route scores this move.

        Resolved once (the run's configuration cannot change mid-run) and
        logged; ``_batch_route_reason`` records why the per-walker path was
        kept."""
        if self._batch_route_state is None:
            self._batch_route_state = ok = self._resolve_batched_route()
            if ok:
                logger.info("[PSD_BATCH] walker-batched ACA scoring: ON")
            else:
                logger.info(
                    "[PSD_BATCH] walker-batched ACA scoring: OFF (%s)",
                    self._batch_route_reason,
                )
        return self._batch_route_state

    def _resolve_batched_route(self) -> bool:
        def _no(reason):
            self._batch_route_reason = reason
            return False

        if not self.psd_batch:
            return _no("PSD_BATCH=0")
        backend = self.sensitivity_backend
        if not isinstance(backend, CompositeSensitivityBackend):
            return _no(
                f"sensitivity backend is {type(backend).__name__}, "
                "not CompositeSensitivityBackend"
            )
        if backend._instrument_basis_cache is None:
            return _no("instrument basis cache disabled")
        if not issubclass(backend.instrument_component_cls, InstrumentNoise):
            return _no("instrument component is not an InstrumentNoise")
        if isinstance(backend.basis_settings, CoarseWDMSettings):
            # The batched scorer gathers the FINE residual buffer and scores
            # it against ``C`` directly. On a coarse grid the model lives on
            # (Nf_active, Ncoarse) while the residual is still fine, and the
            # coarse likelihood is a different quantity anyway (it reads the
            # precomputed sufficient statistic + frozen WS dof). The coarse
            # routes in ``compute_log_like`` handle this case and run first;
            # this is the second guard so the ordering there is not the only
            # thing standing between a coarse run and a wrong likelihood.
            return _no("coarse WDM likelihood uses its own batched route")
        if not isinstance(backend.basis_settings, (FDSettings, WDMSettings)):
            return _no(
                f"unsupported domain {type(backend.basis_settings).__name__}"
            )
        try:
            if not self._batch_instrument_probe()._linear_in_noise_levels():
                return _no("instrument model is not linear in the noise levels")
        except Exception as exc:  # duck-typed test backends etc.
            return _no(f"instrument probe failed: {exc!r}")
        acs = getattr(self, "acs", None)
        for attr in (
            "linear_data_arr", "gpu_splits", "split_map", "nchannels",
            "end_shape", "shape_sens", "_run_per_split", "_split_rows",
        ):
            if not hasattr(acs, attr):
                return _no(f"ACA lacks {attr}")
        if tuple(acs.shape_sens) != (3, 3):
            return _no(f"channel structure {tuple(acs.shape_sens)} != (3, 3)")
        try:
            settings = acs.settings
        except Exception:
            settings = backend.basis_settings
        if getattr(settings, "frequency_layer_mask", None) is not None:
            # ``apply_frequency_layer_mask`` is a no-op on active-grid
            # arrays (``arr.shape[-2] == Nf_active`` — exactly what the
            # ACA's data / invC slices are); it only compresses FULL-grid
            # arrays, and the batched route does not reproduce that.
            if tuple(acs.end_shape) != tuple(settings.basis_shape_active):
                return _no(
                    "frequency_layer_mask would compress the data "
                    f"(end_shape {tuple(acs.end_shape)} != active basis "
                    f"{tuple(settings.basis_shape_active)})"
                )
        try:
            for w in range(len(acs)):
                if getattr(acs[w], "likelihood_source_only", False):
                    return _no("a container is likelihood_source_only")
        except Exception as exc:
            return _no(f"container check failed: {exc!r}")
        return True

    def _batch_instrument_probe(self):
        """A representative instrument component sharing the backend's cache.

        Built exactly the way :meth:`CompositeSensitivityBackend._build_matrix`
        builds the per-walker components (same class, same kwargs, same
        basis-cache dict, a model of the same class), so ``probe._bases``
        warms/reads the SAME cache entry the per-walker route uses and
        ``probe._linear_in_noise_levels()`` answers for the real per-walker
        models."""
        backend = self.sensitivity_backend
        model = backend.instrument_model_cls(
            1.0, 1.0, backend._orbits, f"{backend.model_name}:psd_batch_probe"
        )
        component_kwargs = dict(
            tdi_generation=backend.tdi_generation,
            model=model,
            fill_nans=backend.instrument_fill_nans,
            **backend.instrument_component_kwargs,
        )
        component_kwargs["basis_cache"] = backend._instrument_basis_cache
        return backend.instrument_component_cls(**component_kwargs)

    def _batch_device_runtime(self, device) -> dict:
        """Per-device walker-independent pieces of the batched build.

        Instrument bases (via the shared basis cache), stochastic model
        classes + modulation matrices, and the extra components' (fixed)
        covariances -- built ONCE per device inside its context and kept for
        the run (allocate-once rule); never pickled."""
        rt = self._batch_runtime.get(device)
        if rt is not None:
            return rt
        backend = self.sensitivity_backend
        settings = self._backend_settings_for_device(device)
        probe = self._batch_instrument_probe()
        B_oms, B_acc = probe._bases(settings)
        galfor_probe = GalacticForeground(
            foreground_params=(0.0,) * 5,
            modulation=backend.galfor_modulation,
            tdi_generation=backend.tdi_generation,
            stochastic_fn=backend.galfor_stochastic_fn,
        )
        sgwb_probe = SGWB(
            sgwb_params=(0.0,),
            stochastic_fn=backend.sgwb_stochastic_fn,
            tdi_generation=backend.tdi_generation,
        )
        rt = dict(
            settings=settings,
            B_oms=B_oms,
            B_acc=B_acc,
            Xsens=_XYZ_ELEMENT_SENS[backend.tdi_generation][0],
            galfor_fn=check_stochastic(backend.galfor_stochastic_fn),
            galfor_mod=galfor_probe.time_modulation(settings),
            sgwb_fn=check_stochastic(backend.sgwb_stochastic_fn),
            sgwb_mod=sgwb_probe.time_modulation(settings),
            extra_covs=[
                comp.covariance(settings) for comp in backend.extra_components
            ],
        )
        self._batch_runtime[device] = rt
        return rt

    def _batched_stochastic_mag(self, settings, param_rows, stoch_fn, Xsens, xp):
        """Per-walker stochastic-only magnitudes, batched.

        WDM: evaluates each walker's (cheap, 1-D) spectrum on the fold's
        rFFT bins, then runs ONE batched fold through the existing
        ``FDSignal.wdmtransform(is_psd=True)`` with the walkers stacked on
        the channel axis -- the exact per-walker ``_wdm_layer_psd`` math
        (see :func:`~lisatools.sensitivity.get_sensitivity`), one gather /
        window-square / sum for the whole batch. Returns ``(nb, Nf_active)``
        columns (broadcast along time downstream, mirroring the stationary
        ``repeat``). FD: per-row :func:`get_sensitivity` (1-D, cheap),
        stacked to ``(nb, Nf)``."""
        if isinstance(settings, WDMSettings):
            f_full = xp.fft.rfftfreq(settings.N, settings.data_dt)
            df = float(f_full[1] - f_full[0])
            idx = settings.fold_frequency_indices
            f_idx = f_full[idx]
            psd_full = None
            for r in range(len(param_rows)):
                psd_active = xp.asarray(
                    Xsens.get_Sn(
                        f_idx,
                        stochastic_params=tuple(
                            np.asarray(param_rows[r], dtype=float)
                        ),
                        stochastic_function=stoch_fn,
                        include_instrument=False,
                    )
                )
                if psd_full is None:
                    psd_full = xp.zeros(
                        (len(param_rows), f_full.shape[0]),
                        dtype=psd_active.dtype,
                    )
                psd_full[r, idx] = psd_active
            psd_fd = FDSignal(
                psd_full,
                FDSettings(
                    f_full.shape[0], df, force_backend=settings.backend
                ),
            )
            folded = xp.real(
                psd_fd.wdmtransform(settings=settings, is_psd=True)
            )
            col = folded[:, :, 0]
            # get_sensitivity's fill_nans=0.0 on the folded PSD
            col[xp.isnan(col)] = 0.0
            return col

        from ...sensitivity import get_sensitivity

        rows = [
            xp.asarray(
                get_sensitivity(
                    settings,
                    sens_fn=Xsens,
                    stochastic_params=tuple(np.asarray(p, dtype=float)),
                    stochastic_function=stoch_fn,
                    include_instrument=False,
                    fill_nans=0.0,
                )
            )
            for p in param_rows
        ]
        return xp.stack(rows)

    @staticmethod
    def _add_modulated_mag(C, mag, mod, settings, xp):
        """Add a stochastic component's ``mag x modulation`` into ``C`` in place.

        The batched twin of :meth:`SeparableComponent.covariance` for the
        magnitude-on-every-element components (galfor / sgwb): ``C`` is
        ``(nch, nch, nb, *basis)``; ``mag`` is ``(nb, Nf)`` for the WDM
        basis (broadcast along the trailing time axis) or ``(nb, *basis)``
        generally."""
        from ...sensitivity import _basis_time_axis

        nch = C.shape[0]
        nbasis = C.ndim - 3
        if mag.ndim - 1 < nbasis:
            mag = mag[..., None]  # WDM column -> broadcast over time
        if mod is None:
            for i in range(nch):
                for j in range(nch):
                    C[i, j] += mag
            return
        mod = xp.asarray(mod)
        if mod.ndim == 2:
            for i in range(nch):
                for j in range(nch):
                    C[i, j] += mag * mod[i, j]
            return
        if mod.ndim == 3:
            time_axis = _basis_time_axis(settings)
            if time_axis is None:
                raise ValueError(
                    "time-varying modulation on a domain with no time axis"
                )
            ntime = mod.shape[2]
            shape = [1] * (1 + nbasis)  # broadcast against (nb, *basis)
            shape[1 + time_axis] = ntime
            for i in range(nch):
                for j in range(nch):
                    C[i, j] += mag * mod[i, j].reshape(shape)
            return
        raise ValueError("modulation must be 2D (nch,nch) or 3D (nch,nch,Ntime).")

    def _score_split_batch(
        self, split, device, walkers, psd_phys, galfor_phys, sgwb_phys
    ) -> np.ndarray:
        """Score one shard's walker batch: batched build + batched likelihood.

        Must be called inside ``device``'s context. ``walkers`` are the
        (unique) physical walker indices owned by ``split``; params are
        physical-basis host rows."""
        acs = self.acs
        xp = acs.xp
        rt = self._batch_device_runtime(device)
        settings = rt["settings"]
        nb = len(walkers)
        basis_nd = rt["B_oms"].ndim - 2

        # instrument: C[..] = Soms_d^2 * B_oms + Sa_a^2 * B_acc, walker axis
        # broadcast (the exact per-walker InstrumentNoise.base_covariance
        # fast path, LISAModel stores the squared levels)
        Soms2 = xp.asarray(np.asarray(psd_phys[:, 0], dtype=float) ** 2)
        Sa2 = xp.asarray(np.asarray(psd_phys[:, 1], dtype=float) ** 2)
        bshape = (1, 1, nb) + (1,) * basis_nd
        C = (
            rt["B_oms"][:, :, None] * Soms2.reshape(bshape)
            + rt["B_acc"][:, :, None] * Sa2.reshape(bshape)
        )
        if galfor_phys is not None:
            mag = self._batched_stochastic_mag(
                settings, galfor_phys, rt["galfor_fn"], rt["Xsens"], xp
            )
            self._add_modulated_mag(C, mag, rt["galfor_mod"], settings, xp)
        if sgwb_phys is not None:
            mag = self._batched_stochastic_mag(
                settings, sgwb_phys, rt["sgwb_fn"], rt["Xsens"], xp
            )
            self._add_modulated_mag(C, mag, rt["sgwb_mod"], settings, xp)
        for cov in rt["extra_covs"]:
            C += cov[:, :, None]

        # batched det/inv + the same sanitization as _setup_det_and_inv
        detC, invC = _mat3x3_det_inv(C, xp)
        C_nbytes = C.nbytes
        # C is only consumed by the PSD_DEBUG_CHECKS validation; with the
        # checks off, drop our reference here so the allocator can reuse the
        # covariance stack for the (temporary-heavy) sanitization + likelihood.
        sens_for_check = C if psd_debug_checks_enabled() else None
        del C

        # Unconditional in-place masked writes: the `if bool(xp.any(...))`
        # probes these replaced were a full-array reduction AND a
        # device->host sync per scoring call, to save a single extra pass we
        # pay anyway. Same values either way (non-finite invC -> 0, non-finite
        # detC -> 1); ``copyto`` keeps it allocation-free.
        xp.copyto(invC, 0.0, where=~xp.isfinite(invC))
        xp.copyto(detC, 1.0, where=~xp.isfinite(detC))

        # residual gather straight off the shard's contiguous buffer (the
        # per-container views are rebinds INTO it, so it is authoritative)
        split_rows = np.asarray(acs.gpu_splits[split], dtype=int)
        order = np.argsort(split_rows)
        intra = order[
            np.searchsorted(split_rows[order], np.asarray(walkers, dtype=int))
        ]
        shaped = acs.linear_data_arr[split].reshape(
            (len(split_rows), int(acs.nchannels)) + tuple(acs.end_shape)
        )
        # `intra` is the identity whenever the batch is the whole shard (the
        # common case): take a view instead of copying the full residual.
        if intra.shape[0] == shaped.shape[0] and np.array_equal(
            intra, np.arange(shaped.shape[0])
        ):
            res_batch = shaped
        else:
            res_batch = shaped[intra]

        if (device, nb) not in self._batch_mem_logged:
            self._batch_mem_logged.add((device, nb))
            logger.info(
                "[PSD_BATCH] device %s: batched build for %d walkers -- "
                "C %.1f MB + invC %.1f MB + detC %.1f MB + residual gather "
                "%.1f MB (total ~%.1f MB)",
                device, nb, C_nbytes / 1e6, invC.nbytes / 1e6,
                detC.nbytes / 1e6, res_batch.nbytes / 1e6,
                (C_nbytes + invC.nbytes + detC.nbytes + res_batch.nbytes) / 1e6,
            )

        ll = batched_residual_full_source_and_noise_likelihoods(
            res_batch, sens_for_check, invC, detC, settings
        )
        return asnumpy(ll)

    def _score_walker_batch(
        self, batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
    ) -> np.ndarray:
        """Score one one-row-per-walker batch through the batched route.

        Applies the branch transforms per row on the host (identical to the
        per-walker :meth:`_build_sensitivity_for_walker` seam), then runs one
        :meth:`_score_split_batch` per populated shard through the ACA's
        ``_run_per_split`` (threaded per split on multi-GPU)."""
        acs = self.acs
        xp = acs.xp
        batch = np.asarray(batch, dtype=int)
        walkers = np.asarray(walker_inds_keep, dtype=int)[batch]

        def _phys(transform_fn, coords):
            if coords is None:
                return None
            return np.stack([
                np.asarray(
                    self._to_physical(transform_fn, coords[int(r)]),
                    dtype=float,
                )
                for r in batch
            ])

        psd_phys = _phys(self.psd_transform_fn, psd_coords)
        galfor_phys = _phys(self.galfor_transform_fn, galfor_coords)
        sgwb_phys = _phys(self.sgwb_transform_fn, sgwb_coords)

        out = np.empty(len(batch), dtype=float)
        split_to_pos = acs._split_rows(walkers)

        def _worker(split, positions):
            positions = np.asarray(positions, dtype=int)
            device = None if acs.gpus is None else int(acs.gpus[split])
            with device_context(xp, device):
                ll = self._score_split_batch(
                    split, device, walkers[positions], psd_phys[positions],
                    None if galfor_phys is None else galfor_phys[positions],
                    None if sgwb_phys is None else sgwb_phys[positions],
                )
            out[positions] = np.asarray(ll, dtype=float)

        acs._run_per_split(
            _worker, split_to_pos,
            run_threaded=True if self._run_threaded else None,
        )
        return out

    def _compute_log_like_batched(
        self, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
    ) -> np.ndarray:
        """Batched scoring of all surviving rows, grouped one-row-per-walker.

        The grouping mirrors the per-walker route's batching loop -- here it
        bounds the batched build's memory at ``nwalkers`` covariance stacks
        per group (not ``ntemps * nwalkers``), NOT because anything is
        installed on the containers (nothing is)."""
        n = int(np.asarray(walker_inds_keep).shape[0])
        tmp_logl = np.empty(n, dtype=float)
        remaining = np.arange(n)
        while remaining.size:
            _, first = np.unique(walker_inds_keep[remaining], return_index=True)
            batch = remaining[np.sort(first)]
            tmp_logl[batch] = self._score_walker_batch(
                batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
            )
            remaining = np.setdiff1d(remaining, batch)
        return tmp_logl

    def compute_log_like(self, coords, inds=None, logp=None, supps=None, branch_supps=None):
        """Compute the PSD/galfor branch log-likelihood.

        Hybrid routing (stft_tof merge): when the C++ XYZ kernel fast path is
        available (FD/STFT basis, single shard), score the proposals directly
        through ``XYZSensitivityBackend.compute_log_like``. Otherwise fall
        back to the dev-side, domain-agnostic route: install the proposed PSD
        (and optional galactic foreground) into each walker's
        :class:`AnalysisContainer` ``sens_mat`` and read the per-walker
        likelihood from the shared :class:`AnalysisContainerArray` (works for
        FD / STFT / WDM alike).

        Args:
            coords: Branch-keyed dict of coordinates from the current state.
            inds: Branch-keyed dict of leaf occupancy flags (unused here).
            logp: Optional pre-computed log prior; computed if ``None``.
            supps: Eryn supplemental information; ``walker_inds`` selects
                which physical walker each row maps to.
            branch_supps: Branch-supplemental dict (unused here).

        Returns:
            Tuple ``(logl, blobs)`` matching the eryn signature.
        """
        if logp is None:
            logp = self.compute_log_prior(coords, inds=inds, supps=supps, branch_supps=branch_supps)

        assert logp is not None
        logl = np.full_like(logp, -1e300)

        logp_keep = ~np.isinf(logp)
        if not np.any(logp_keep):
            warnings.warn("All points entering likelihood have a log prior of minus inf.")
            return logl, None

        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        # ``walker_inds`` is broadcast (ntemps, nwalkers) — flatten and mask
        # to the rows that survived the prior cut. ``BranchSupplemental``'s
        # ``__getitem__`` expects an integer/slice index, so reach into
        # ``.holder`` to fetch the named entry.
        walker_inds_all = np.asarray(supps.holder["walker_inds"]).reshape(logp.shape)
        walker_inds_keep = walker_inds_all[logp_keep].astype(int)

        # full noise model per row: sampled branches from the proposal,
        # fixed branches from the cold-row snapshot (per walker)
        merged = self._merged_noise_rows(coords, logp_keep, walker_inds_keep)
        psd_coords = merged["psd"]
        has_galfor = "galfor" in merged
        galfor_coords = merged.get("galfor")
        has_sgwb = "sgwb" in merged
        sgwb_coords = merged.get("sgwb")

        if self._kernel_fast_path_available(has_sgwb=has_sgwb):
            # stft_tof fast path: C++ shared-memory kernel.
            if has_galfor:
                input_args = [psd_coords, galfor_coords]
            else:
                input_args = [psd_coords]

            supps_keep = supps[logp_keep]
            logl[logp_keep] = self.psd_log_like(input_args, supps=supps_keep, **self.psd_kwargs)

            self.prev_logl = logl.copy()
            return logl, None

        # The coarse-WDM routes come FIRST: they score against the coarse
        # sufficient statistic, which the PSD_BATCH route below knows nothing
        # about (it gathers the FINE residual buffer). Both are guarded, but
        # order is the first line of defence.
        if self._galfor_subband_fast_path_available():
            stat = self.acs.flatten()[0].coarse_stats
            tmp_logl = self._compute_galfor_subband_loglike(
                stat, walker_inds_keep, galfor_coords
            )
            if tmp_logl is not None:
                logl[logp_keep] = tmp_logl
                self.prev_logl = logl.copy()
                return logl, None

        if self._coarse_batch_fast_path_available():
            from ...coarsewdm import coarse_wdm_log_likelihood_batch

            stat = self.acs.flatten()[0].coarse_stats
            tmp_logl = np.empty(walker_inds_keep.shape[0], dtype=float)
            remaining = np.arange(walker_inds_keep.shape[0])
            while remaining.size:
                _, first = np.unique(walker_inds_keep[remaining], return_index=True)
                batch = remaining[np.sort(first)]
                covariances = self._build_covariance_batch(
                    batch,
                    walker_inds_keep,
                    psd_coords,
                    galfor_coords if has_galfor else None,
                    sgwb_coords if has_sgwb else None,
                )
                tmp_logl[batch] = np.asarray(
                    coarse_wdm_log_likelihood_batch(stat, covariances)
                )
                remaining = np.setdiff1d(remaining, batch)
            logl[logp_keep] = tmp_logl
            self.prev_logl = logl.copy()
            return logl, None

        if self._batched_route_ready():
            # PSD_BATCH walker-batched route (tier 3): one batched covariance
            # build + one batched likelihood per shard per scoring group.
            # Nothing is installed on the containers, so there is no
            # sens-mat snapshot/restore and no linear_psd_arr repack here
            # (the publish repack in propose() still runs and is what the
            # rest of the run reads).
            logl[logp_keep] = self._compute_log_like_batched(
                walker_inds_keep,
                psd_coords,
                galfor_coords if has_galfor else None,
                sgwb_coords if has_sgwb else None,
            )
            self.prev_logl = logl.copy()
            return logl, None

        # Cache and restore the per-walker sensitivity matrix so we don't
        # corrupt the state seen by other moves. After all proposals are
        # scored we put each AC's original sens_mat back; the caller (the
        # ``propose`` loop) reinstalls the accepted PSD onto the ACs.
        # There are ntemps*nwalkers proposal rows but only nwalkers ACs, and
        # ``walker_inds`` maps every rung of walker w onto acs[w]. Scoring all
        # rows in one pass would let each rung overwrite the previous one's
        # sens_mat, and the single batched ``likelihood()`` would then hand
        # every rung of a walker the SAME value (the last one written) --
        # the cold rung scored with a hot rung's parameters, so the Metropolis
        # ratio compares unrelated likelihoods and accepts nearly everything.
        # Score in batches instead, each holding a walker at most once.
        original_sens = {}
        tmp_logl = np.empty(walker_inds_keep.shape[0], dtype=float)
        try:
            remaining = np.arange(walker_inds_keep.shape[0])
            while remaining.size:
                # first remaining row for each distinct walker -> one batch
                _, first = np.unique(walker_inds_keep[remaining], return_index=True)
                batch = remaining[np.sort(first)]
                # Snapshot the originals serially before any building: the
                # threaded path overwrites acs[w].sens_mat, so this must not
                # race with it (and it is a bare attribute read anyway).
                for row in batch:
                    w = int(walker_inds_keep[row])
                    if w not in original_sens:
                        original_sens[w] = self.acs[w].sens_mat
                self._build_batch(
                    batch,
                    walker_inds_keep,
                    psd_coords,
                    galfor_coords if has_galfor else None,
                    sgwb_coords if has_sgwb else None,
                )
                # See _skip_linear_psd_repack -- WARNING there before adding a
                # WDM C++ likelihood kernel. acs.likelihood() reads
                # ac.sens_mat.invC, not the buffer, so on a no-kernel run this
                # repack is ~16 ms/call of memcpy nobody reads.
                if not self._skip_linear_psd_repack:
                    self.acs.reset_linear_psd_arr()
                walker_ll = asnumpy(np.asarray(self.acs.likelihood()))
                tmp_logl[batch] = walker_ll[walker_inds_keep[batch].astype(int)]
                remaining = np.setdiff1d(remaining, batch)
            logl[logp_keep] = tmp_logl
        finally:
            for w, sens in original_sens.items():
                self.acs[w].sens_mat = sens
            if not self._skip_linear_psd_repack:
                self.acs.reset_linear_psd_arr()

        self.prev_logl = logl.copy()

        return logl, None

    def compute_log_prior(self, branches_coords, *args, **kwargs):
        """Sum the per-branch log priors over THIS move's sampled branches.

        Fixed (non-sampled) noise branches are constants during this move's
        Metropolis-Hastings updates, so their priors cancel in the acceptance
        ratio and must not be summed here (a split move using the joint prior
        would double-count them against another noise move).

        Args:
            branches_coords: Branch-keyed dict of coordinates.

        Returns:
            ``(ntemps, nwalkers)`` array of prior log probabilities.
        """
        # wait to get ntemps, nwalkers
        logp = None
        for key in self.NOISE_BRANCHES:
            if key not in branches_coords:
                continue
            if self.sampled_branches is not None and key not in self.sampled_branches:
                continue
            ntemps, nwalkers, _, ndim = branches_coords[key].shape
            if logp is None:
                logp = np.zeros((ntemps, nwalkers))

            logp[:] += (
                self.priors[key]
                .logpdf(branches_coords[key].reshape(-1, ndim))
                .reshape(ntemps, nwalkers)
            )
        return logp

    def _propose_delayed_acceptance(self, model, state):
        """One stretch repeat under two-stage (delayed) acceptance.

        Stage 1 reuses eryn's stretch accept VERBATIM as the surrogate
        screen: the working state's log-likes are swapped to the coarse
        values (so eryn's ``logP(x)`` is the tempered coarse posterior) and
        ``model`` carries :meth:`compute_coarse_log_like` (so ``logP(y)`` is
        too); the prior and the stretch factor therefore enter exactly once,
        in stage 1. Stage 2 corrects the stage-1 survivors with

            log alpha_2 = beta * [(Lf(y) - Lc(y)) - (Lf(x) - Lc(x))]

        (priors and factors cancel), reverting rejects to their pre-proposal
        coordinates. The returned state's ``log_like`` is FINE on every row
        — the invariant the temperature swaps and the sub-state bookkeeping
        rely on. Fine evaluations are spent only on stage-1 survivors (the
        prior-mask trick: non-survivor rows enter the fine callback with a
        ``-inf`` prior and are skipped).

        The independent stage-2 uniforms come from ``model.random`` AFTER
        eryn's stage-1 draws, so a fixed seed reproduces the full kernel.
        """
        # Pre-proposal snapshot: coords + FINE lls + priors. eryn's propose
        # may mutate ``state`` in place, so this copy is the revert source
        # and the alpha_2 reference. is_eryn_state_input: the working state
        # is the module-ladder tmp_state (dict-constructed, no sub-states);
        # the plain-GFState copy path would trip over its absent
        # sub_state_bases, and the snapshot needs none of that machinery.
        state_before = GFState(state, copy=True, is_eryn_state_input=True)
        fine_x = np.asarray(state_before.log_like)

        coarse_x = self.compute_coarse_log_like(
            state.branches_coords,
            logp=state.log_prior,
            supps=state.supplemental,
        )[0]

        # Stage 1: eryn stretch accept on the tempered coarse posterior.
        # The temperature control is DETACHED for this call: eryn's propose
        # ends with temper_comps (the per-repeat identity swap), which would
        # permute rungs between the stage-1 accept and the stage-2
        # correction. The swap is re-applied below, after the fine invariant
        # is restored — so it swaps on FINE values, as every other swap does.
        state.log_like[:] = coarse_x
        _tc = self.temperature_control
        self.temperature_control = None
        try:
            new_state, accepted1 = super(PSDMove, self).propose(model, state)
        finally:
            self.temperature_control = _tc
        acc = np.asarray(accepted1, dtype=bool)

        if not acc.any():
            new_state.log_like[:] = fine_x
            self._da_debug_last = None
            new_state = self.temperature_control.temper_comps(new_state)
            return new_state, acc

        # Stage 2: fine correction on the survivors only.
        coarse_y = np.asarray(new_state.log_like).copy()
        survivor_logp = np.where(acc, np.asarray(new_state.log_prior), -np.inf)
        fine_y = self.compute_log_like(
            new_state.branches_coords,
            logp=survivor_logp,
            supps=new_state.supplemental,
        )[0]
        betas = np.asarray(self.temperature_control.betas)[:, None]
        log_alpha2 = betas * ((fine_y - coarse_y) - (fine_x - coarse_x))
        u2 = model.random.uniform(size=acc.shape)
        with np.errstate(divide="ignore"):
            keep = acc & (np.log(u2) < log_alpha2)
        revert = acc & ~keep

        for key in new_state.branches:
            new_state.branches[key].coords[revert] = state_before.branches[
                key
            ].coords[revert]
        new_state.log_prior[revert] = np.asarray(state_before.log_prior)[revert]
        # FINE invariant restored: survivors carry their fresh fine values,
        # everything else keeps the pre-proposal fine values.
        new_state.log_like[:] = np.where(keep, fine_y, fine_x)

        self._da_debug_last = {
            "accepted_stage1": acc,
            "keep": keep,
            "fine_x": fine_x,
            "fine_y": fine_y,
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "log_alpha2": log_alpha2,
            "coords_final_prewap": np.array(
                new_state.branches["psd"].coords, copy=True
            )
            if "psd" in new_state.branches
            else None,
        }
        # The per-repeat identity swap eryn would have run inside propose,
        # now on the restored FINE values.
        new_state = self.temperature_control.temper_comps(new_state)
        return new_state, keep

    def run_move(self, move_i, model, state):
        """Run one stretch-move iteration and (optionally) a tempering swap.

        Args:
            move_i: Iteration index used to schedule the periodic
                :attr:`permute_every` temperature swap.
            model: Eryn ``Model`` object.
            state: Current sampler state.

        Returns:
            Tuple ``(new_state, accepted)``.
        """
        if (
            self.coarse_sidecar_active
            and self.coarse_runtime.mode == "delayed_acceptance"
        ):
            new_state, accepted = self._propose_delayed_acceptance(model, state)
        else:
            new_state, accepted = super(PSDMove, self).propose(model, state)

        # in-model bookkeeping: eryn returns (ntemps, nwalkers) acceptances and
        # every walker is proposed once per call, so the per-temperature deltas
        # are the row sums / the walker count. Tallies are reset per propose()
        # and written into the sub-states there (see delta_counter_names).
        acc = np.asarray(accepted)
        acc = acc.reshape(acc.shape[0], -1)
        if self._tally_in_model_accepted is not None:
            self._tally_in_model_accepted += acc.sum(axis=-1).astype(int)
            self._tally_in_model_proposed += acc.shape[-1]

        # Tempering, addremove-style two-tier cadence (2026-08-12): a cheap
        # IDENTITY-paired naive swap fires EVERY repeat -- valid within a
        # walker because every rung of walker j scores against walker j's own
        # residual, so the dbeta*dlogl factor is exact and costs no
        # likelihood evaluations -- and the walker-PERMUTING fancy swap
        # (re-scores the swapped params against the DESTINATION walkers'
        # residuals through compute_log_like) fires only every
        # ``permute_every`` repeats. Previously NO swap ran between fancy
        # events, so the ladder sat unexchanged for the other repeats.
        do_fancy = (move_i % self.permute_every == 0)
        x = new_state.branches_coords
        logl = new_state.log_like
        logp = new_state.log_prior
        branch_supps = new_state.branches_supplemental
        supps = new_state.supplemental

        logP = self.compute_log_posterior(logl, logp)
        x, logP, logl, logp, inds, blobs, supps, branch_supps = (
            self.temperature_control.temperature_swaps(
                x,
                logP,
                logl,
                logp,
                supps=supps,
                branch_supps=branch_supps,
                # search_approx swaps on the surrogate (it optimizes that
                # target); off/delayed_acceptance swap on the FINE values —
                # identity swaps read the fine state.log_like and fancy swaps
                # re-score through the fine callback (plan-2 §3.2).
                compute_log_like=getattr(self, "_swap_like_fn", None)
                or self.compute_log_like,
                compute_log_prior=self.compute_log_prior,
                fancy_swap=do_fancy,
                permute_here=do_fancy,
            )
        )

        for name in x:
            new_state.branches[name].coords[:] = x[name][:]
            new_state.branches[name].branch_supplemental = branch_supps[name]

        new_state.log_like[:] = logl[:]
        new_state.log_prior[:] = logp[:]
        new_state.supplemental = supps

        # swap bookkeeping: TemperatureControl refreshes swaps_accepted on
        # every call and holds swaps_proposed fixed at nwalkers per rung.
        tc = self.temperature_control
        if self._tally_swaps_accepted is not None:
            sa = getattr(tc, "swaps_accepted", None)
            sp = getattr(tc, "swaps_proposed", None)
            if sa is not None:
                self._tally_swaps_accepted += np.asarray(sa).ravel().astype(int)
            if sp is not None:
                self._tally_swaps_proposed += np.asarray(sp).ravel().astype(int)

        return new_state, accepted

    def run_move_for_loop(self, model, state, num_repeats):
        """Run :meth:`run_move` ``num_repeats`` times sequentially."""
        for i in tqdm(
            range(num_repeats), desc="psd update",
            disable=not getattr(self, "progress", False),
        ):
            state, accepted = self.run_move(i, model, state)
        return state, accepted

    def run_move_max_likelihood(self, model, state):
        """Repeat :meth:`run_move_for_loop` until the max log-like plateaus.

        Used in search-style runs (``max_logl_mode=True``). The loop counts
        consecutive iterations during which the cold-chain max log-likelihood
        no longer increases and exits after ``num_checks`` such iterations.
        """
        num_checks = 5
        num_so_far = 0
        max_logl = -np.inf
        changed_once = False
        while num_so_far < num_checks:
            state, accepted = self.run_move_for_loop(model, state, self.num_repeats)

            if state.log_like[0].max() != max_logl and not np.isinf(max_logl):
                changed_once = True

            if state.log_like[0].max() > max_logl:
                max_logl = state.log_like[0].max()
                num_so_far = 0
            else:
                if changed_once:
                    num_so_far += 1

        return state, accepted

    def propose(self, model, state):
        """Propose a noise-model update and refresh per-walker sensitivity matrices.

        Builds a temporary :class:`GFState` containing only THIS move's
        sampled noise branches (non-sampled noise branches are frozen at
        their cold rows and merged into every likelihood evaluation), runs
        the inner stretch-move loop, then writes the accepted coordinates
        back into a copy of ``state`` and refreshes each walker's
        sensitivity matrix in :attr:`acs`.

        Returns:
            Tuple ``(new_state, accepted)``.
        """
        _t_prop0 = time.perf_counter()
        # GB_PROP_TIMING_SYNC=1 attributes device time to the span that
        # launched it rather than the one that forces the sync (same
        # semantics as the GB moves). Resolved defensively: PSDMove has no
        # guaranteed ``xp``/``backend``, and a timer must never be able to
        # take down a propose.
        _sync = None
        if os.environ.get("GB_PROP_TIMING_SYNC", "0") == "1":
            _xp = getattr(self, "xp", None)
            if getattr(getattr(self, "backend", None), "uses_cupy", False) and _xp is not None:
                _sync = _xp.cuda.runtime.deviceSynchronize
        self._prop_timer = _tm = _ProposeTimer(sync_fn=_sync)

        model_branches = self._noise_model_branches(state)
        noise_branches = self._resolve_sampled(state)

        # cold-row agreement between the main state and the noise-model
        # branches' sub-states — fixed inputs included, they feed the
        # likelihood too (GF_SUBSTATE_CHECK=0 disables)
        self._check_substate_consistency(state, model_branches)
        engine_ntemps = state.log_like.shape[0]

        # freeze the non-sampled noise branches at their cold rows for the
        # duration of this proposal block (consumed by _merged_noise_rows)
        self._fixed_noise_coords = {
            key: np.asarray(state.branches_coords[key][0, :, 0]).copy()
            for key in model_branches
            if key not in noise_branches
        }
        self._prepare_fixed_component_covariances()
        if self.coarse_sidecar_active:
            # Correctness-first lifecycle (plan-2 §4.3): every noise proposal
            # block re-reads every walker's CURRENT residual — a source move
            # may have run since the last block. Residual epochs are a
            # recorded later optimization, deliberately not built yet.
            self.coarse_runtime.refresh_P(
                [self.acs[w] for w in range(len(self.acs))]
            )
            self._prepare_fixed_component_covariances_coarse()

        # The working ensembles are the SUB-STATES' tempered branches (this
        # move's module ladder); the main state carries only the engine's
        # cold chain.
        tmp_branches_coords = {
            key: self._work_branch(state, key).coords for key in noise_branches
        }

        shapes = {key: tmp_branches_coords[key].shape[:2] for key in noise_branches}
        if len(set(shapes.values())) != 1:
            raise ValueError(
                "Noise branches sampled by ONE move must share (ntemps, nwalkers); "
                f"got {shapes}. Match the branches' ntemps knobs "
                "(PSD_NTEMPS / GALFOR_NTEMPS / SGWB_NTEMPS) or sample them "
                "with separate moves."
            )

        # move-local supplemental at the MODULE ladder shape (the main
        # state's supplemental is engine-shaped)
        nt_mod, nwalkers_mod = tmp_branches_coords[noise_branches[0]].shape[:2]

        # per-iteration acceptance deltas, accumulated across the repeat block
        # by run_move and written into each sampled branch's sub-state below.
        self._tally_in_model_proposed = np.zeros(nt_mod, dtype=int)
        self._tally_in_model_accepted = np.zeros(nt_mod, dtype=int)
        self._tally_swaps_proposed = np.zeros(max(nt_mod - 1, 0), dtype=int)
        self._tally_swaps_accepted = np.zeros(max(nt_mod - 1, 0), dtype=int)
        tmp_supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(nwalkers_mod), (nt_mod, 1))},
            base_shape=(nt_mod, nwalkers_mod),
            copy=True,
        )

        # module-ladder acceptance tracking (the stage combine-move setter
        # presets an engine-shaped array, so re-shape on first use)
        _acc = getattr(self, "_accepted", None)
        if _acc is None or np.shape(_acc) != (nt_mod, nwalkers_mod):
            self.accepted = np.zeros((nt_mod, nwalkers_mod))

        tmp_state = GFState(tmp_branches_coords, copy=True, supplemental=tmp_supps)

        # ensuring it is up to date. Should not change anything.
        before_vals = model.analysis_container_arr.likelihood().copy()

        # Coarse-sidecar mode dispatch (plan-2 §3.2/§6):
        #   off               — fine everywhere (the historical path).
        #   search_approx     — the whole inner loop optimizes the coarse
        #                       surrogate (proposals, working state, swaps);
        #                       NOT a posterior sampler. The end-of-propose
        #                       publication below still recomputes the fine
        #                       cold state, so the next move sees fine only.
        #   delayed_acceptance— the working state's log_like invariant stays
        #                       FINE; eryn scores stage-1 proposals with the
        #                       coarse callback and _propose_delayed_acceptance
        #                       applies the stage-2 fine correction. Swaps are
        #                       fine-valued by construction.
        _mode, _init_like_fn, _model_like_fn, self._swap_like_fn = (
            self._resolve_mode_like_fns()
        )

        tmp_state.log_prior = self.compute_log_prior(tmp_branches_coords)
        tmp_state.log_like = _init_like_fn(
            tmp_branches_coords, logp=tmp_state.log_prior, supps=tmp_state.supplemental
        )[0]
        self.starting_now = False

        tmp_model = Model(
            state,
            _model_like_fn,
            self.compute_log_prior,
            self.temperature_control,
            model.map_fn,
            model.random,
        )

        with _tspan(_tm, "sample"):
            if self.max_logl_mode:
                tmp_state, accepted = self.run_move_max_likelihood(tmp_model, tmp_state)

            else:
                tmp_state, accepted = self.run_move_for_loop(tmp_model, tmp_state, self.num_repeats)

        # CHECK THIS STATE SETUP
        new_state = GFState(state, copy=True)

        for key in noise_branches:
            if key not in tmp_state.branches:
                continue
            # full-ladder write into the sub-state (shared-memory view),
            # cold row mirrored into the main (engine) state. Only the
            # SAMPLED branches are written: each noise branch's sub-state is
            # owned by the move that samples it (full-noise-model tempered
            # lnL of ITS ensemble + ITS move's adapted ladder).
            self._work_branch(new_state, key).coords[:] = tmp_state.branches[key].coords[:]
            self._sync_cold_row(new_state, key)
            sub = (new_state.sub_states or {}).get(key)
            if sub is not None and getattr(sub, "tempered_initialized", False):
                sub.log_like[:] = tmp_state.log_like[:]
                sub.log_prior[:] = tmp_state.log_prior[:]
                if hasattr(sub, "betas") and sub.betas is not None:
                    sub.betas[:] = self.temperature_control.betas
                # acceptance deltas for this iteration (the backend zeroes
                # them after each save, so these are per-iteration counts).
                # rj_* stay zero: this move never changes leaf count.
                sub.in_model_proposed[:] = self._tally_in_model_proposed
                sub.in_model_accepted[:] = self._tally_in_model_accepted
                if sub.swaps_proposed.size:
                    sub.swaps_proposed[:] = self._tally_swaps_proposed
                    sub.swaps_accepted[:] = self._tally_swaps_accepted

        # the engine state keeps only the cold row (row 0 rewritten from the
        # refreshed ACS below). The cold prior is the FULL noise-model sum so
        # it doesn't flip-flop between partial priors as split moves run.
        new_state.log_like[0] = tmp_state.log_like[0]
        new_state.log_prior[0] = self._cold_noise_log_prior(new_state)

        # Publish the accepted (cold-row) noise model onto every walker's
        # AnalysisContainer. Same per-shard dispatch as the scoring builds:
        # each walker's build runs on its owning device, split groups run
        # concurrently under run_threaded (multi-GPU), serially otherwise.
        nwalkers = len(self.acs)

        def _publish_one(w):
            w = int(w)
            psd_params = new_state.branches_coords["psd"][0, w, 0]
            if "galfor" in new_state.branches_coords:
                galfor_params = new_state.branches_coords["galfor"][0, w, 0]
            else:
                galfor_params = None
            if "sgwb" in new_state.branches_coords:
                sgwb_params = new_state.branches_coords["sgwb"][0, w, 0]
            else:
                sgwb_params = None

            with _tspan(_tm, "sens_refresh"):
                new_sens = self._build_sensitivity_for_walker(
                    w, psd_params, galfor_params, sgwb_params
                )
            self.acs[w].sens_mat = new_sens
            if self.debug and w == self.debug_plot_walker:
                # accepted params of the plotted walker, for the
                # instrument-only reference curve in the debug overlay
                self._debug_walker_params = (
                    w, psd_params, galfor_params, sgwb_params
                )

        self._run_rows_per_split(
            _publish_one, np.arange(nwalkers), np.arange(nwalkers)
        )

        # NOT gated by _skip_linear_psd_repack, deliberately. This is the one
        # repack that PUBLISHES the accepted noise model to the rest of the
        # run, and it fires once per propose (~16 ms) rather than once per
        # scoring batch (~295x per iteration) -- so skipping it would buy
        # ~0.15% while opening the buffer to staleness across moves. Keeping
        # it means the only window where linear_psd_arr is stale is INSIDE
        # compute_log_like's scoring loop, where no other move can observe it.
        self.acs.reset_linear_psd_arr()
        after_vals = self.acs.likelihood()

        if self.debug:
            if self._debug_step % self.debug_every == 0:
                self._debug_plot_psd_vs_residual()
            self._debug_step += 1

        new_state.log_like[0] = after_vals
        # eryn-facing acceptance uses the engine (cold-chain) shape
        _tm.count("walkers", len(self.acs))
        _tm.count("branches", len(noise_branches))
        logger.info("[PSD_TIMING %s] %s", getattr(self, "name", "psd"),
                    _tm.report(time.perf_counter() - _t_prop0,
                               top=("sample", "sens_refresh", "rows_per_split")))
        self._prop_timer = None
        return new_state, np.asarray(accepted)[:engine_ntemps]


    # ------------------------------------------------------------------
    # DCGA (multi-device) kernel path — absorbed from the legacy
    # MultiGPUPSDMove in the 2026-07 multi-GPU pass (one move structure).
    # ------------------------------------------------------------------

    def _psd_log_like_dcga(self, x: list, supps=None, **sens_kwargs):
        """Kernel fast path through the DCGA's per-device replicas."""
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        wi = supps["walker_inds"]

        psd_pars, galfor_pars = self.transform_coords(x, return_cupy=False)

        data_index_all = np.asarray(wi).astype(np.int32)

        positions_per_split, data_intra_index_per_split, _ = self.dcga.unpack_indices(data_index_all)
        coords_per_split = self.dcga.unpack_coords(positions_per_split, (psd_pars, galfor_pars))

        data_intra_index_per_split, coords_per_split = self.dcga.place_on_device(
            items=(data_intra_index_per_split, coords_per_split)
        )

        likelihood_args_per_split = self.dcga._loop_operation(
            operation=self.prepare_likelihood_inputs,
            operation_args_per_split=coords_per_split,
            positions_per_split=positions_per_split,
        )

        ll = self.dcga.compute_psd_likelihood(
            positions_per_split,
            data_intra_index_per_split,
            data_intra_index_per_split,
            likelihood_args_per_split,
            likelihood_kwargs={'run_async': self.run_async},
            run_threaded=self.run_threaded,
        )

        # now check if any knot position is not too close together
        if self.sensitivity_backend.use_splines:
            invalid_knots_mask = np.zeros(psd_pars.shape[0], dtype=bool)
            for i, likelihood_args in enumerate(likelihood_args_per_split):
                if likelihood_args is None:
                    continue
                spline_knots_position = (
                    likelihood_args[-2].get()
                    if hasattr(likelihood_args[-2], "get")
                    else likelihood_args[-2]
                )

                invalid_knots_mask[positions_per_split[i]] = np.any(
                    np.diff(10**spline_knots_position, axis=2) < self.tolerance, axis=(0, 2)
                )
            ll[invalid_knots_mask] = -1e300

        self.dcga.free_gpu_memory()
        return ll


class MultiGPUPSDMove(PSDMove):
    """Deprecated alias — :class:`PSDMove` is ONE structure that takes
    ``dcga=`` directly (2026-07 multi-GPU pass). This shim maps the legacy
    ``MultiGPUPSDMove(dcga, priors, ...)`` signature onto it.
    """

    def __init__(self, dcga: DomainComputationGroupArray, priors, *args,
                 run_async: bool = False, run_threaded: bool = False,
                 **kwargs):
        warnings.warn(
            "MultiGPUPSDMove is deprecated: construct "
            "PSDMove(acs, priors, dcga=...) directly (one move structure).",
            DeprecationWarning, stacklevel=2,
        )
        PSDMove.__init__(
            self, dcga.acs, priors, *args, dcga=dcga,
            run_async=run_async, run_threaded=run_threaded, **kwargs,
        )
