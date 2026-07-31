"""Galactic-binary specialized stretch / RJ moves and supporting infrastructure."""

from __future__ import annotations

import os
import time
import logging
import warnings
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from inspect import Attribute
from types import ModuleType
from typing import Optional, Union, Tuple

import numpy as np
import numpy

from eryn.state import BranchSupplemental
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N, get_fdot
from scipy import stats

from ... import sensitivity
from ...detector import sangria
from ...utils.constants import *
from ...analysiscontainer import (
    AnalysisContainer,
    AnalysisContainerArray,
    BandView,
    band_gpu_assignment,
)
# DataResidualArray is deprecated; AnalysisContainer now accepts DomainBase children
# directly (FDSignal / WDMSignal / TDSignal / STFTSignal).
from ...domains import DomainSettingsBase, FDSettings, WDMSettings
from ...sensitivity import SensitivityMatrixBase
from ...utils.parallelbase import LISAToolsParallelModule
from ...utils.device import device_context, pin_main_device
from ...utils.utility import asnumpy
from gbgpu.gb_likelihood import (
    BandLikelihoodEngine,
    FDBandLikelihoodEngine,
    SwapLLResult,
    WDMBandLikelihoodEngine,
    make_band_likelihood_engine,
)
from .globalfitmove import GFCombineMove, GlobalFitMove
from ..priors.gbpriors import get_fdot_mojito

# -*- coding: utf-8 -*-


try:
    import cupy as cp
    import cupy
    gpu_available = True
except ModuleNotFoundError:
    import numpy as cp

    gpu_available = False

from eryn.moves import GroupStretchMove, Move, StretchMove
from eryn.moves.multipletry import get_mt_computations, logsumexp
from eryn.paraensemble import ParaEnsembleSampler
from eryn.priors import ProbDistContainer, UniformDistribution
from eryn.utils import PeriodicContainer
from eryn.utils.utility import groups_from_inds

from ...diagnostic import inner_product
from ...sampling.prior import FullGaussianMixtureModel, GBPriorWrap
from ...utils.utility import get_array_module, get_groups_from_band_structure, searchsorted2d_vec
from ..state import GFState, ensure_leaf_cap_fields

__all__ = ["GBSpecialStretchMove"]

logger = logging.getLogger(__name__)

class _NoOpMempool:
    """CPU stand-in for ``cupy.get_default_memory_pool()`` — calls become no-ops."""

    def free_all_blocks(self):
        return


# ``_to_numpy`` was the file-local cupy/numpy-agnostic ``.get()`` helper.
# Use the central :func:`lisatools.utils.utility.asnumpy` instead.
_to_numpy = asnumpy


# The MPI worker entry points (gb_search_func / fit_each_leaf /
# gb_refit_func) that fanned CPU distribution fitting out to spare ranks
# were removed with the move->rank dispatch (parallel-resources plan P3);
# GPU GMM fitting (vec_fit_gmm_min_bic) replaced them.


from dataclasses import dataclass

from eryn.state import Branch
from eryn.utils import TransformContainer


# Band-level infrastructure now lives in gbbands.py; re-exported here so
# existing ``from ...gbspecialstretch import Buffer / BandSorter`` imports
# keep working.
from .gbbands import (
    BandScheduler,
    BandSorter,
    Buffer,
    SubBandBuffer,
    _RoutedBandEngine,
    pack_special_index,
    return_x,
    unpack_special_index,
)


class _ProposeTimer:
    """Accumulating wall-clock stage timer for one GB ``propose()`` call.

    Localizes where a proposal spends its time (GPU-efficiency diagnosis):
    stages are accumulated with :meth:`span` and reported as a single
    sorted INFO line at the end of the propose. Overhead is a pair of
    ``perf_counter`` calls per span, so it stays on by default.

    On a CuPy backend the numbers are HOST wall time per stage. Because
    kernel launches are asynchronous, device work is attributed to the
    stage that *forces* the sync (the next ``_to_numpy`` / ``.item()`` /
    explicit synchronize). Set ``GB_PROP_TIMING_SYNC=1`` to synchronize the
    device at every span boundary instead — slightly slower overall, but
    each stage then carries exactly its own kernel time. Either view is
    diagnostic: if host time dominates in stages with tiny kernels
    (``inmodel_repeats`` with few cells), the run is launch-overhead-bound
    (too few sub-bands/cells per launch to keep the GPU busy).
    """

    __slots__ = ("stages", "counts", "_sync")

    def __init__(self, sync_fn=None):
        self.stages: dict = {}
        self.counts: dict = {}
        self._sync = sync_fn

    @contextmanager
    def span(self, name: str):
        if self._sync is not None:
            self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._sync is not None:
                self._sync()
            self.stages[name] = self.stages.get(name, 0.0) + (
                time.perf_counter() - t0
            )

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(n)

    def report(self, total: float) -> str:
        # Top-level stages only: nested spans (buffer_build inside
        # run_proposal, ...) are reported but excluded from the
        # tracked/untracked accounting via the ``_TOP`` list.
        top = (
            "sorter_build", "friend_index", "resid_open_close", "ll_checks",
            "run_proposal", "run_tempering", "write_back", "sorter_rebuild",
            "band_info", "ll_inject_final", "ll_inject_drift", "mempool_free",
        )
        items = sorted(self.stages.items(), key=lambda kv: -kv[1])
        tracked = sum(v for k, v in self.stages.items() if k in top)
        parts = [f"{k}={v:.3f}s" for k, v in items]
        cparts = [f"{k}={v}" for k, v in sorted(self.counts.items())]
        return (
            f"total={total:.3f}s tracked={tracked:.3f}s "
            f"untracked={max(total - tracked, 0.0):.3f}s | "
            + " ".join(parts)
            + (" | " + " ".join(cparts) if cparts else "")
        )


def _tspan(tm, name: str):
    """Timer span or no-op when the propose-level timer is absent."""
    return tm.span(name) if tm is not None else nullcontext()


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialBase(GlobalFitMove, GroupStretchMove, Move, LISAToolsParallelModule):
    """Base class for GB-specific stretch / reversible-jump moves.

    Combines :class:`GlobalFitMove`, :class:`eryn.moves.GroupStretchMove`,
    :class:`Move`, and :class:`LISAToolsParallelModule` so each GB move can
    use try-force rejection, optional phase maximization, and GPU-resident
    band-aware buffers (:class:`Buffer`, :class:`BandSorter`).

    ``sequential_parity_repeats`` (class flag, default ``False``): when
    ``True``, :meth:`_run_in_model_repeats` runs each repeat as TWO
    sequential half-sweeps split by walker parity — eryn's
    :class:`~eryn.moves.RedBlueMove` split structure, with the complement
    always at its CURRENT state — so every half-sweep is an invariant
    kernel and ``num_repeat_proposals`` is a cost knob, not a bias knob.
    ``False`` keeps the single full-batch sweep per repeat, which is
    correct for the group-stretch / info-matrix proposals whose complement
    structures (friend table, Cholesky) are block-start snapshots by
    design. Set ``True`` only on moves whose ``in_model_proposal`` reads
    its complement from ``band_sorter.coords``
    (:class:`VGBSpecialStretchMove`).

    # TODO/DOCS: full argument list — many constructor kwargs are passed
    through unmodified to ``GroupStretchMove``. Intended use is via the
    concrete subclasses :class:`GBSpecialStretchMove`,
    :class:`GBSpecialRJPriorMove`,
    :class:`GBSpecialRJSerialSearchMCMC`, and
    :class:`GBSpecialRJRefitMove`.

    Args:
        gb: :class:`gbgpu.GBGPU` instance.
        priors: :class:`ProbDistContainer` for in-model GB parameters.
        start_freq_ind: Inclusive starting index into the global ``f_arr``
            (FD path); ignored on the WDM path.
        data_length: Length of the data array per channel (FD path); the
            WDM path sizes its per-band buffers from
            ``WDMSettings.Nf_active`` / ``Nt_active`` instead.
        acs: :class:`AnalysisContainerArray` used for SETUP ONLY: the
            basis-domain settings on ``acs.settings`` drive every
            domain-dependent choice (FD vs WDM) and the initial parent
            binding. It is not stored -- at run time everything reads
            from / fills into the ACA that arrives with the model in
            :meth:`propose` (re-binding if it changed).
        band_edges: Frequency-band edges.
        band_N_vals: Per-band waveform sample counts.
        gpu_priors: Branch-keyed GPU-resident priors.
        waveform_kwargs: Forwarded to ``gb`` waveform calls.
        parameter_transforms: :class:`TransformContainer` for GBs.
        snr_lim: Optional SNR cut.
        rj_proposal_distribution: Distribution used to draw RJ proposals.
        is_rj_prop: Marks this move as a reversible-jump proposal.
        num_repeat_proposals: Inner repeat count per call.
        name: Move name (used for logging and bookkeeping).
        branch_name: Eryn branch this move operates on. Default ``"gb"``;
            the VGB move passes ``"vgb"``. All state / prior / periodic
            dict keys go through this.
        use_info_mat_proposal: When ``False``, the in-model repeats never
            build the information-matrix Cholesky proposal (pure stretch).
            Default ``True`` (GB behavior).
        swap_on_in_model: Run the band-temperature swap stage even when this
            move is not an RJ proposal. Default ``False`` (GB reference
            recipe runs swaps on its RJ moves); the fixed-dimensional VGB
            move sets ``True`` since it has no RJ move to carry the swaps.
        preserve_leaf_identity: Write accepted coordinates back to each
            source's ORIGINAL leaf index instead of re-indexing leaves
            densely in frequency order. ``None`` (default) auto-resolves to
            ``True`` when ``f0`` is a per-leaf transform fill (leaf i is a
            specific physical source, e.g. VGBs) and ``False`` otherwise.
        use_prior_removal: If ``True``, draw RJ proposals from the prior.
        phase_maximize: If ``True``, marginalize over phase in the
            likelihood.
        gpus: GPU device list for this move (intra-node knob).
        num_band_preload: Number of bands preloaded per call.
        run_swaps: Whether to run band-temperature swaps.
        max_data_store_size: Cap on the per-iteration data store size.
        force_backend: Optional backend override.
        gb_wdm_comp: Optional :class:`gbgpu.gbcomps.GBWDMComputations`
            instance. Required when ``acs.settings`` is a
            :class:`~lisatools.domains.WDMSettings`; ignored otherwise.
    """

    # See the class docstring; True only on VGBSpecialStretchMove.
    sequential_parity_repeats = False

    @property
    def xp(self) -> Union[ModuleType, numpy , cupy]:
        """Active array module (NumPy or CuPy) for this move."""
        return self.backend.xp

    def __init__(
        self,
        gb: GBGPU,
        priors,
        start_freq_ind,
        data_length,
        acs,
        band_edges,
        band_N_vals,
        gpu_priors,
        *args,
        waveform_kwargs={},
        parameter_transforms: Optional[TransformContainer] = None,
        snr_lim=1e-10,
        rj_proposal_distribution=None,
        is_rj_prop=False,
        num_repeat_proposals=100,
        name=None,
        branch_name="gb",
        use_info_mat_proposal=True,
        swap_on_in_model=False,
        preserve_leaf_identity=None,
        use_prior_removal=False,
        phase_maximize=False,
        gpus=[],
        num_band_preload=20000,
        wdm_band_slab_layers=None,
        wdm_slab_guard_layers=1,
        run_swaps=True,
        max_data_store_size=6000,
        force_backend=None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        orbits=None,
        tdi_config=None,
        t_ref=0.0,
        search_kwargs=None,
        stretch_probability=0.5,
        band_units=2,
        jump_factor=0.005,
        leaf_cap_start=None,
        leaf_cap_min_iters=50,
        leaf_cap_ll_nsigma=3.0,
        leaf_cap_require_occupancy=True,
        leaf_cap_update=True,
        sighet_refresh_every=0,
        sighet_refresh_dphase=0.5,
        sighet_refresh_min_beta=0.1,
        sighet_trust_dlna=1.5,
        sighet_trust_dphase=0.5,
        sighet_trust_snr_c=30.0,
        sighet_trust_dlna_min=0.3,
        sighet_anchor_check=False,
        sighet_drift_check=False,
        debug_seq_pick="first",
        debug=False,
        debug_plot_dir="./gf_output/gb_debug/",
        debug_plot_walker=0,
        debug_plot_band=None,
        **kwargs,
    ):
        # return_gpu is a kwarg for the stretch move
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        GlobalFitMove.__init__(self, name=name)
        Move.__init__(self, *args, return_gpu=True, **kwargs)

        # Stretch-move state. GroupStretchMove.__init__ is not chained (the
        # MRO would double-init Move), so the pieces its get_proposal /
        # get_new_points read are set explicitly here.
        self.a = float(kwargs.get("a", 2.0))
        self.nfriends = int(kwargs.get("nfriends", 32))
        self.return_gpu = True
        self.use_gpu = self.backend.uses_cupy

        self.force_backend = force_backend
        self.gpus = gpus
        self.gpu_priors = gpu_priors
        self.num_repeat_proposals = num_repeat_proposals
        # ``n_subbands`` is the user-facing alias for the number of
        # (temp, walker, band) cells held in the sub-band buffer at once.
        if kwargs.get("n_subbands") is not None:
            num_band_preload = int(kwargs["n_subbands"])
        self.num_band_preload = self.n_subbands = num_band_preload
        # Task-b: narrow per-band WDM slab extent (layers). None = full active
        # band (bit-identical to pre-task-b); 0 = auto-size (band span +
        # 2*(leakage+guard)); N>0 = explicit. ``wdm_slab_guard_layers`` is the
        # adjustable guard used by the auto-size. Forwarded to every
        # BandSorter -> SubBandBuffer.
        self.wdm_band_slab_layers = wdm_band_slab_layers
        self.wdm_slab_guard_layers = wdm_slab_guard_layers
        self.band_preload_size = self.max_data_store_size = max_data_store_size
        self.use_prior_removal = use_prior_removal
        self.has_setup_group = False

        # GB-sampler verification instrumentation. When ``debug`` is on, the
        # ``_debug_*`` hooks below run band residual round-trip / get_ll
        # consistency checks and dump begin/middle/end band plots at the real
        # operation sites in ``run_proposal``. Off by default -> the hooks
        # early-return, so the production path is untouched.
        self.debug = bool(debug)
        self.debug_plot_dir = debug_plot_dir
        # Plot selection: only the chosen (walker, band) cell is plotted --
        # ONE figure per plotted step with a panel per temperature -- instead
        # of a PNG for every (temp, walker, band) the proposal touches.
        # ``debug_plot_band=None`` resolves to the central band at plot time.
        self.debug_plot_walker = int(debug_plot_walker)
        self.debug_plot_band = (None if debug_plot_band is None
                                else int(debug_plot_band))
        # Which of the traced cell's sources the sequence figures follow:
        # "first" = whatever the first pick round selects (default);
        # "loudest" = wait for the round that picks the cell's max-amplitude
        # source; a number (mHz) = wait for the source nearest that f0.
        self.debug_seq_pick = str(debug_seq_pick).lower()
        self._dbg_plot_counter = 0

        # for key in priors:
        #     if not isinstance(priors[key], ProbDistContainer) and not isinstance(priors[key], GBPriorWrap):
        #         raise ValueError(
        #             "Priors need to be eryn.priors.ProbDistContainer object."
        #         )

        # Per-band progressive leaf cap (search mode). ``leaf_cap_start``
        # arms the machinery: fresh states get ``band_leaf_cap[:] =
        # leaf_cap_start`` and RJ births into a band already holding
        # ``cap[b]`` alive leaves are prior-forbidden (curr_logp = -inf) at
        # EVERY temperature -- the cap is a truncation of the prior on the
        # per-band leaf count, so it must be temperature-uniform for the
        # per-band tempering swaps to exchange states of a common support.
        # Convergence is judged on the COLD chain only (see
        # ``_update_band_leaf_caps``); ``leaf_cap_update`` marks the single
        # RJ move per iteration that advances the cap state.
        self.leaf_cap_start = leaf_cap_start
        self._leaf_cap_enabled = leaf_cap_start is not None
        self.leaf_cap_min_iters = int(leaf_cap_min_iters)
        self.leaf_cap_ll_nsigma = float(leaf_cap_ll_nsigma)
        self.leaf_cap_require_occupancy = bool(leaf_cap_require_occupancy)
        self.leaf_cap_update = bool(leaf_cap_update)
        self._band_leaf_cap = None

        # Sig-het reference policy (ALL in-model proposals): the heterodyne
        # reference is built ONCE per repeat block, against the source-free
        # residual, and held FIXED for the whole block --
        # ``sighet_refresh_every=0`` (the default) disables the legacy
        # mid-block drift refresh entirely. The block's addback cross-check
        # keeps the STORED ll exact regardless; the fixed reference only
        # bounds the per-repeat MH deltas by the heterodyne linearization
        # over one block's walk. Set ``sighet_drift_check=True``
        # (GB_SIGHET_DRIFT_CHECK=1) to LOG the end-of-block drift metric
        # (accumulated carrier-phase drift + amplitude ratio vs the
        # expansion point) without changing the sampling -- the audit knob
        # for that approximation.
        #
        # ``sighet_refresh_every=N > 0`` re-enables the legacy per-source
        # mid-block refresh (drift test every N repeats, only offenders
        # rebuilt, ll_ref re-based; beta-gated below). Diagnostic /
        # comparison use. Inert when the engine's setup hook is a no-op
        # (chunked-het / FD).
        self.sighet_refresh_every = int(sighet_refresh_every)
        self.sighet_refresh_dphase = float(sighet_refresh_dphase)
        self.sighet_drift_check = bool(sighet_drift_check)
        # Refresh only where the scoring error matters: below this beta a
        # stale reference's ll error is beta-suppressed in the acceptance
        # exponent, while every refresh costs a full reference rebuild --
        # the dominant sig-het expense (~seconds/setup on production
        # grids; hot junk sources otherwise trip the drift test at nearly
        # every checkpoint).
        self.sighet_refresh_min_beta = float(sighet_refresh_min_beta)
        # Sig-het TRUST REGION (prior = -inf outside): in-model candidates
        # whose PHYSICAL-amplitude ratio vs the block's heterodyne anchor
        # exceeds ``sighet_trust_dlna`` e-folds (or whose carrier-phase
        # drift exceeds ``sighet_trust_dphase`` rad) are rejected before
        # scoring. The expansion is only trusted near its reference
        # (measured: exact to ~3e-6 through |dlnA| ~ 8 on a clean source,
        # but in-run weak-source offenders corrupt at large excursions),
        # and a detectable source's posterior never comes near the gate
        # (lnA width ~ 1/SNR). MH-valid as a proposal-support restriction:
        # the anchor is the block-start state (re-anchored on refresh), so
        # the current point always sits inside its own region and the
        # indicator is symmetric in (x, y). 0 disables. Inert on
        # chunked-het / FD (no sig-het reference active).
        self.sighet_trust_dlna = float(sighet_trust_dlna)
        self.sighet_trust_dphase = float(sighet_trust_dphase)
        # PER-SOURCE SNR SCALING of the amplitude gate. The sig-het
        # truncation error is RELATIVE to the source's own template power,
        # so the ABSOLUTE lnL error a walker can accrue at the gate
        # boundary scales with h_h ~ SNR^2: a uniform gate lets an SNR-80
        # source carry an O(1) absolute error at |dlnA| = 1.5 while an
        # SNR-3 source's error there is negligible. Scaling the gate as
        #
        #     dlnA_max(i) = clip(C / snr_ref(i), dlna_min, sighet_trust_dlna)
        #
        # makes the absolute error ceiling roughly uniform across the
        # catalogue. snr_ref = sqrt(h_h) at the block anchor -- free from
        # the ll_ref evaluation. Statistically the scaled gate never binds
        # for detectable sources: their lnA posterior width is ~1/SNR, so
        # C = 30 sits ~30 sigma out; weak sources keep the global cap,
        # where their absolute error is small. C = 0 reverts to the
        # uniform gate (back-compat); the whole gate still disables via
        # sighet_trust_dlna = 0. Refresh re-anchors snr_ref along with
        # the reference.
        self.sighet_trust_snr_c = float(sighet_trust_snr_c)
        self.sighet_trust_dlna_min = float(sighet_trust_dlna_min)
        # ANCHOR CHECK (debug, GB_SIGHET_ANCHOR_CHECK=1): at block start,
        # after the reference build and ll_ref evaluation, score the SAME
        # anchor coordinates through the exact engine and compare. At the
        # anchor the heterodyne ratio is exactly 1, so any discrepancy is
        # an ANCHOR-LEVEL offset in the reference/coefficients for that
        # source (window/slab truncation, geometry) — cleanly separated
        # from candidate-displacement error, which the end-of-block audit
        # measures. Found via the 2026-07-30 CPU offender log: a source at
        # |dll| = 7.6 with dlnA = 3e-3 can only be an anchor offset.
        # Costs one exact batched call + one reference rebuild per block.
        self.sighet_anchor_check = bool(sighet_anchor_check)

        self.priors = priors
        self.gb = gb
        # Optional WDM-domain likelihood object. Constructed once by the
        # user (typically a gbgpu.gbcomps.GBWDMComputations) and threaded
        # through to BandSorter -> Buffer -> WDMBandLikelihoodEngine when
        # the analysis container's DomainSettingsBase is a WDMSettings.
        # Stays None on the FD path; the FD engine path doesn't touch it.
        self.gb_wdm_comp = gb_wdm_comp
        self.stop_here = True
        self.run_swaps = run_swaps

        if self.backend.uses_cupy:
            self.mempool = self.xp.get_default_memory_pool()
        else:
            # TODO: add a NoOpMempool setup to the backend itself
            self.mempool = _NoOpMempool()

        self.band_edges = band_edges
        self.num_bands = len(band_edges) - 1
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.parameter_transforms = parameter_transforms
        # NOTE: the constructor ACA (``acs``) is used for SETUP ONLY (domain
        # dispatch, shapes, initial parent binding). It is NOT stored: every
        # run-time fill / likelihood targets the ACA that arrives with the
        # model in :meth:`propose`, which re-binds the parent engine whenever
        # that ACA differs from the one currently bound.

        self._configure_domain(acs)
        self.phase_maximize = phase_maximize

        # Analytic amplitude maximisation of RJ births (search heuristic that
        # pairs with phase_maximize). The template is linear in amplitude, so
        # a birth's add-delta ``s*d_h - 0.5*s^2*h_h`` is maximised at the ML
        # scale ``s* = d_h/h_h``; rescaling the sampled amplitude/distance to
        # s* lets a well-placed intrinsic draw (e.g. an F-stat proposal on a
        # real peak) reach ``delta ~ 0.5*SNR^2`` instead of being rejected on
        # a random prior-drawn amplitude. Like phase-max this relaxes strict
        # reversibility, so it is a search move only. Defaults to follow
        # ``phase_maximize``; ``GB_RJ_AMP_MAXIMIZE`` overrides.
        _amp_env = os.environ.get("GB_RJ_AMP_MAXIMIZE")
        self.rj_amp_maximize = (
            bool(int(_amp_env)) if _amp_env is not None else bool(phase_maximize)
        )

        # F-stat distance-birth proposal (step 2, supersedes the d_h/h_h pin
        # above). Centers the birth on the F-statistic's 4-parameter maximized
        # amplitude ``A_max`` (Jaranowski-Krol inversion of ``a = M^-1 N``),
        # converts A_max -> distance via (f0, Mc), and DRAWS the birth distance
        # from a lognormal about that center with width set by the F-stat SNR
        # (``ln dist ~ N(ln dist*, 1/SNR)``); phi0/iota/psi are centered on
        # their F-stat maxima. It is a real proposal (density enters the RJ
        # factor), so unlike the pin it is detailed-balance-valid.
        #
        # A_max comes from the NEW domain-native F-stat comps
        # (``GBFDComputations.get_fstat_ll_fd`` / ``GBWDMComputations.
        # get_fstat_ll_wdm`` -> per-binary (N, M)); the Jaranowski-Krol
        # maximization is a SINGLE shared routine
        # (``fstat_maximized_extrinsics``) applied identically to either
        # domain's (N, M). Never the legacy SharedMemoryGBGPU
        # ``gb.get_fstat_ll`` (FD-only; floods a WDM buffer). Defaults to
        # follow ``rj_amp_maximize`` (on under the search config); when on it
        # replaces the d_h/h_h pin. ``GB_RJ_FSTAT_DIST_BIRTH=0`` falls back to
        # that pin. NOTE: the (N,M)->F path is validated (the proposal grid
        # finds the real source), but the A_max inversion on those pieces is
        # new -- verify births land at sane distances on the first GPU run.
        _fdb_env = os.environ.get("GB_RJ_FSTAT_DIST_BIRTH")
        self.rj_fstat_dist_birth = (
            bool(int(_fdb_env)) if _fdb_env is not None
            else bool(self.rj_amp_maximize)
        )
        self._log_dist_range_cache = None

        self.snr_lim = snr_lim

        self.band_edges = self.xp.asarray(self.band_edges)

        self.rj_proposal_distribution = rj_proposal_distribution
        self.is_rj_prop = is_rj_prop or (self.rj_proposal_distribution is not None)

        # if self.is_rj_prop:
        #     if (self.num_repeat_proposals != 1):
        #         print("Adjusting repeat proposals to 1 for RJ.")

        #     self.num_repeat_proposals = 1

        # setup N vals for bands
        self.band_N_vals = self.xp.asarray(band_N_vals)

        self.num_proposals = 0
        self.search_kwargs = search_kwargs
        # In-model proposal mix: probability of drawing the band-aware group
        # stretch instead of the info-matrix Cholesky jump per repeat round.
        # The default in_model_proposal() is the overridable hook consuming
        # this; subclass it for other proposal components.
        self.stretch_probability = float(stretch_probability)
        # Band parity stride: 2 = odds/evens (minimum separation); larger
        # values give wider separation between simultaneously-updated bands.
        self.band_units = max(1, int(band_units))
        # Info-matrix jump scale (Gaussian draw through the Cholesky factor).
        self.jump_factor = float(jump_factor)
        self._fdot_scale = 1e-16

        # Parent binding: config-only FD comp + move-level engine. A user-
        # supplied gb_fd_comp is honored as-is; otherwise one is built from
        # the ACA's FDSettings -- data holders are passed to it at call
        # time, so ACA changes only rewire the engine (see _bind_parent_acs,
        # re-invoked from propose()).
        self.transform_fn = self.parameter_transforms

        # Branch + sampled-basis roles, derived from the transform container
        # (never hardcoded index literals): the GB reference basis keeps its
        # historical columns (f0@1, fdot@2, phi0@3) because that is what its
        # input_basis says, and a reduced basis (e.g. the 5D VGB basis with
        # f0/sky as per-leaf fills) resolves to its own columns / to None.
        self.branch_name = branch_name
        self.use_info_mat_proposal = bool(use_info_mat_proposal)
        self.swap_on_in_model = bool(swap_on_in_model)
        if self.transform_fn is not None and hasattr(self.transform_fn, "input_basis"):
            _ib = list(self.transform_fn.input_basis)
            self._f0_col = _ib.index("f0") if "f0" in _ib else None
            self._phi0_col = _ib.index("phi0") if "phi0" in _ib else None
            self._fdot_col = next(
                (_ib.index(_k) for _k in ("fdot", "Mc") if _k in _ib), None
            )
            # 9th column of the fdot_astro ratio basis (None otherwise).
            self._fdot_astro_col = (
                _ib.index("fdot_astro_ratio") if "fdot_astro_ratio" in _ib
                else None
            )
        else:
            # legacy GB layout when no container is supplied
            self._f0_col, self._fdot_col, self._phi0_col = 1, 2, 3
            self._fdot_astro_col = None
        # Per-leaf fill metadata (Eryn per-leaf fill_dict): position of f0
        # among the fill keys + the (nleaves, n_fill) value table, for band
        # assignment when f0 is not a sampled column.
        self._per_leaf_fill = getattr(self.transform_fn, "n_leaf_fills", None) is not None
        self._f0_fill_col = None
        if self._per_leaf_fill:
            _fill_keys = list(self.transform_fn.original_fill_dict[0].keys())
            if "f0" in _fill_keys:
                self._f0_fill_col = _fill_keys.index("f0")
        if self._f0_col is None and self._f0_fill_col is None:
            raise ValueError(
                f"{type(self).__name__}: 'f0' must be either a sampled column of the "
                "transform input basis or a per-leaf fill key (band machinery needs it)."
            )
        if preserve_leaf_identity is None:
            preserve_leaf_identity = self._f0_col is None
        self.preserve_leaf_identity = bool(preserve_leaf_identity)
        # Whether propose() builds the cold-chain frequency friend table for
        # the group stretch (subclasses with their own partner scheme skip it).
        self._build_friend_table = True
        self._gb_fd_comp_user_supplied = gb_fd_comp is not None
        self.gb_fd_comp = gb_fd_comp
        self._proposal_orbits = orbits
        self._proposal_tdi_config = tdi_config
        self._t_ref = float(t_ref)
        self._bind_parent_acs(acs)


    def _configure_domain(self, acs) -> None:
        """Derive ``self.fd`` / ``self.df`` / ``self._basis_settings`` from an ACA.

        The band-index math uses ``df = 1 / Tobs`` consistently across FD
        and WDM (FD: equals ``acs.df``; WDM: ``acs.df == layer_df`` differs,
        so we recompute). ``self.fd`` is only meaningful in the FD path.
        """

        # TODO: make this more generic
        if isinstance(acs.settings, FDSettings):
            self.fd = acs.f_arr.copy()
            self.df = float(self.fd[1] - self.fd[0])
        elif isinstance(acs.settings, WDMSettings):
            self.fd = None
            self.df = 1.0 / acs.settings.Tobs
        else:
            raise NotImplementedError(
                f"GBSpecialBase does not support basis domain "
                f"{type(acs.settings).__name__}."
            )
        self._basis_settings = acs.settings

    def _bind_parent_acs(self, acs) -> None:
        """(Re)bind the parent-level engine to ``acs``.

        Every run-time fill / likelihood targets the ACA that arrives with
        the model in :meth:`propose`; the constructor ACA only provides the
        initial binding. Post-2026-07 the FD comp is CONFIG-ONLY
        (``GBFDComputations(fd_settings, t_ref, ...)``) -- data holders are
        passed at get_ll/fill_global time, so an ACA change only refreshes
        the engine wiring here, never any C-side data pointers.
        """
        token = (id(acs), id(acs.linear_data_arr[0]))
        if getattr(self, "_parent_acs_token", None) == token:
            return

        # The parent is itself a window of the global rfft grid starting at
        # ``start_freq_ind``; the engine's bounds mask uses this when the
        # holder itself can't resolve per-row starts.
        self._parent_start_inds = self.xp.full(
            int(acs.acs_total_entries), int(self.start_freq_ind),
            dtype=self.xp.int32,
        )

        if isinstance(acs.settings, FDSettings):
            if self._gb_fd_comp_user_supplied:
                if getattr(self, "_parent_acs_token", None) is not None:
                    logger.warning(
                        f"{self.name}: parent ACA changed but gb_fd_comp was "
                        "user-supplied; keeping the user's comp."
                    )
            elif (
                self.gb_fd_comp is None
                # Rebuild if the grid config drifted (band re-slice etc.);
                # cheap, no data pointers involved.
                or self.gb_fd_comp.df != float(self._basis_settings.df)
                or self.gb_fd_comp.ind_min != int(self._basis_settings.ind_min)
                or self.gb_fd_comp.ind_max != int(self._basis_settings.ind_max)
            ):
                # TODO: check this a little further
                from gbgpu.gbcomps import GBFDComputations

                if self._proposal_tdi_config is None:
                    raise ValueError(
                        "FD-basis GB moves need tdi_config= (and orbits=) to "
                        "build the GBFDComputations comp, or pass "
                        "gb_fd_comp= directly."
                    )
                orbits_in = (
                    self._proposal_orbits
                    if self._proposal_orbits is not None
                    else getattr(self.gb, "orbits", None)
                )
                self.gb_fd_comp = GBFDComputations(
                    self._basis_settings,
                    self._t_ref,
                    N_sparse=int(self.band_N_vals.max()),
                    orbits=orbits_in, tdi_config=self._proposal_tdi_config,
                    force_backend=self.force_backend,
                    d_d=0.0,
                    tdi_type=self.waveform_kwargs.get(
                        "tdi_channel_setup", "XYZ"),
                    nchannels=acs.nchannels,
                )

        # Move-level engine for parent-residual fills (cold-chain
        # open/close). Same domain dispatch as the sub-band buffer's engine;
        # the parent ACA has no per-slot ``min_freq_inds``, so the FD engine
        # falls back to ``start_freq_inds`` (one shared window start per
        # walker row). Routed: a multi-shard parent ACA partitions each
        # fill by the walker's owning GPU split.
        self._likelihood_engine = _RoutedBandEngine(make_band_likelihood_engine(
            self._basis_settings,
            gb=self.gb,
            gb_fd_comp=self.gb_fd_comp,
            gb_wdm_comp=self.gb_wdm_comp,
            nchannels=acs.nchannels,
            tdi_channel_setup=self.waveform_kwargs.get("tdi_channel_setup"),
            df=self.df,
            start_freq_inds=self._parent_start_inds,
            data_length=acs.data_length,
        ))
        self._parent_acs_token = token

    def setup(self, model, branches):
        return

    @classmethod
    def supported_backends(cls):
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def find_friends(self, name, s, s_inds=None, branch_supps=None):
        """Complement points for the band-aware group stretch.

        The friends were drawn (one per proposed source) by
        :meth:`BandSorter.draw_friends` immediately before the
        ``GroupStretchMove.get_proposal`` call in :meth:`in_model_proposal`;
        this override just reshapes them to match ``s``
        ``(1, n_sources, 1, ndim)``.
        """
        friends = self._friends_for_stretch
        assert friends.shape[0] == s.shape[1]
        return friends[None, :, None, :]

    def adjust_sources_in_residual_buffer(
        self, factor, model, band_sorter: BandSorter, *args, **kwargs
    ) -> None:
        """Add or remove sources from the main residual buffer.

        Domain-agnostic: uses ``self._likelihood_engine.fill_template`` to
        work with both FD and WDM domains. The ``factor`` controls whether
        sources are added (``-1``, reducing the residual) or removed (``+1``,
        restoring them to the residual).
        """
        assert isinstance(factor, int) and (factor == -1 or factor == +1)

        subset = band_sorter.get_subset(*args, **kwargs)

        if subset is None or subset.inds.sum().item() == 0:
            return

        params_in = subset.coords_in[subset.inds]
        walkers_in = subset.walker_inds[subset.inds].astype(self.xp.int32)
        N_vals_in = subset.N_vals[subset.inds]

        # FD-specific bounds checks (only meaningful when basis is FD).
        # WDM bounds are checked internally by the engine's layer-indexing.
        if isinstance(self._basis_settings, FDSettings):
            f_bin = (params_in[:, 1] / self.df).astype(int) - int(self.start_freq_ind)
            assert not np.any(
                f_bin + (N_vals_in / 2) > model.analysis_container_arr.data_length
            ), "cold-chain source window exceeds the parent data range"
            assert not np.any(f_bin - (N_vals_in / 2) < 0), (
                "cold-chain source window falls below the parent data range"
            )

        # ``coords_in`` is already in physical units; dispatch via the engine.
        self._likelihood_engine.fill_template(
            model.analysis_container_arr,
            params_in,
            walkers_in,
            N_vals_in,
            factor=factor,
            waveform_kwargs=self.waveform_kwargs,
        )

    def remove_cold_chain_sources_from_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        kwargs["apply_inds"] = True
        self.remove_sources_from_residual(*args, **kwargs)

    def remove_sources_from_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(+1, *args, **kwargs)

    def add_cold_chain_sources_to_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        kwargs["apply_inds"] = True
        self.add_sources_to_residual(*args, **kwargs)

    def add_sources_to_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(-1, *args, **kwargs)

    # ================= GB-sampler verification (debug mode) =================
    # Each hook does ONE piece at the site where that operation happens in
    # run_proposal; all early-return unless ``self.debug`` is set (so the
    # production path is untouched). Together they realize the residual
    # round-trip the sampler relies on: load the neighbour cold-chain residual
    # -> the band buffer holds that walker's signals -> remove the source being
    # proposed -> get_ll -> add it back -> ll, verifying the single-source
    # get_ll is consistent with both the swap_ll the sampler scores and the
    # full residual change, plus begin/middle/end band plots.

    @property
    def _debug_tiles(self) -> bool:
        """Whether the GB_DEBUG per-cell slab snapshots / band-tile plots run.

        The 3x3 sequence figures, the RJ before/after pair, the band null
        log and :meth:`_debug_plot_band` all reshape a cell's residual slab
        into a WDM ``(nchannels, Nf_active, Nt_active)`` tile and render 2-D
        wavelet images, so they only make sense on the WDM basis. On the FD
        basis there is no such tile (the slab is a flat frequency window), so
        these are skipped; the domain-agnostic GB_DEBUG checks
        (get_add_ll/get_removal_ll deltas, the residual add/remove round-trip,
        the removal identity, and the timing report) still run under
        ``self.debug`` regardless of domain.
        """
        return self.debug and isinstance(self._basis_settings, WDMSettings)

    def _debug_cold_chain_residual_loaded(self, model, remainder) -> None:
        """At the cold-chain residual load site: log the neighbour cold-chain
        residual baseline ll for this band unit ("load neighbouring residual")."""
        if not self.debug:
            return
        try:
            ll_np = _to_numpy(model.analysis_container_arr.likelihood())
            logger.info(
                "[GB_DEBUG %s] cold-chain residual loaded (remainder=%s): "
                "sum ll = %.6e over %d walker(s)",
                self.name, remainder, float(np.sum(ll_np.real)), ll_np.size,
            )
        except Exception as e:  # debug-only: never break the sampler
            logger.warning("[GB_DEBUG %s] cold-chain snapshot skipped: %r", self.name, e)

    def _debug_verify_rj_step(self, buffer_obj, params, alive, slots, N_vals,
                              delta_ll, keep, picked, round_i, scheduler) -> None:
        """At the RJ scoring site: independently re-verify the birth/death
        deltas through get_add_ll / get_removal_ll, check the removal
        identity ``<r+h|h> = <r|h> + <h|h>`` by an actual residual
        round-trip, and plot the band on the first rounds."""
        if not self.debug:
            return
        try:
            xp = self.xp
            k = _to_numpy(keep).astype(bool)
            if not k.any():
                return
            births = k & ~_to_numpy(alive)
            deaths = k & _to_numpy(alive)

            if births.any():
                b = xp.asarray(np.where(births)[0])
                check = buffer_obj.get_add_ll(params[b], slots[b], slots[b], N_vals[b])
                lhs = delta_ll[b]
                fin = xp.isfinite(check) & (lhs > -1e290)
                if bool(fin.any()):
                    relmax = float(xp.max(xp.abs(lhs[fin] - check[fin])
                                          / xp.maximum(xp.abs(check[fin]), 1.0)))
                    logger.info(
                        "[GB_DEBUG %s] RJ birth delta vs get_add_ll: max rel %.3e",
                        self.name, relmax,
                    )
                d_h_b = buffer_obj.d_h_out.copy()
                h_h_b = buffer_obj.h_h_out.copy()
                self._debug_residual_round_trip(
                    buffer_obj, params[b], slots[b], N_vals[b], d_h_b, h_h_b
                )

            if deaths.any():
                d = xp.asarray(np.where(deaths)[0])
                check = buffer_obj.get_removal_ll(params[d], slots[d], slots[d], N_vals[d])
                lhs = delta_ll[d]
                fin = xp.isfinite(check) & (lhs > -1e290)
                if bool(fin.any()):
                    relmax = float(xp.max(xp.abs(lhs[fin] - check[fin])
                                          / xp.maximum(xp.abs(check[fin]), 1.0)))
                    logger.info(
                        "[GB_DEBUG %s] RJ death delta vs get_removal_ll: max rel %.3e",
                        self.name, relmax,
                    )
                # Removal identity: restore the template (factor +1), then
                # <r+h|h> must equal <r|h> + <h|h>. Restored in ``finally``.
                d_h_1 = buffer_obj.d_h_out.copy()
                h_h_1 = buffer_obj.h_h_out.copy()
                eng = buffer_obj._likelihood_engine
                params_phys = self.transform_fn.both_transforms(params[d], xp=cp)
                di = slots[d].astype(xp.int32)
                eng.fill_template(buffer_obj, params_phys, di, N_vals[d],
                                  factor=+1, waveform_kwargs=self.waveform_kwargs)
                try:
                    buffer_obj.get_ll(params[d], slots[d], slots[d], N_vals[d])
                    expected = (d_h_1 + h_h_1).real
                    relmax = float(xp.max(xp.abs(buffer_obj.d_h_out.real - expected)
                                          / xp.maximum(xp.abs(expected), 1.0)))
                    logger.info(
                        "[GB_DEBUG %s] removal identity <r+h|h>=<r|h>+<h|h>: max rel %.3e",
                        self.name, relmax,
                    )
                finally:
                    eng.fill_template(buffer_obj, params_phys, di, N_vals[d],
                                      factor=-1, waveform_kwargs=self.waveform_kwargs)

            if round_i in (0, 1):
                map_cpu = (
                    _to_numpy(picked["temp_inds"]),
                    _to_numpy(picked["walker_inds"]),
                    _to_numpy(picked["band_inds"]),
                )
                self._debug_plot_band(buffer_obj, params, slots, N_vals,
                                      delta_ll, map_cpu, keep, round_i,
                                      stage="rj")
        except Exception as e:  # debug-only: never break the sampler
            logger.warning("[GB_DEBUG %s] verify_rj_step skipped: %r", self.name, e)

    def _debug_verify_in_model(self, buffer_obj, curr, new, slots, N_vals,
                               delta_ll, keep, map_cpu, move_i) -> None:
        """At the in-model repeat site: on the first repeat run the residual
        round-trip on the current source (the buffer holds the source-free
        residual) and plot the band at begin/middle/end of the repeats."""
        if not self.debug:
            return
        try:
            if move_i == 0:
                buffer_obj.get_ll(curr, slots, slots, N_vals)
                d_h_c = buffer_obj.d_h_out.copy()
                h_h_c = buffer_obj.h_h_out.copy()
                self._debug_residual_round_trip(
                    buffer_obj, curr, slots, N_vals, d_h_c, h_h_c
                )
            if move_i in (0, self.num_repeat_proposals // 2,
                          max(self.num_repeat_proposals - 1, 0)):
                self._debug_plot_band(buffer_obj, new, slots, N_vals, delta_ll,
                                      map_cpu, keep, move_i, stage="in-model")
        except Exception as e:  # debug-only: never break the sampler
            logger.warning("[GB_DEBUG %s] verify_in_model skipped: %r", self.name, e)

    def _debug_residual_round_trip(self, buffer_obj, params_add, data_index,
                                   swap_N_vals, d_h_a, h_h_a) -> None:
        """Add the proposed template to the band residual (factor=-1), confirm
        the get_ll shift equals -<h|h>, then remove it (factor=+1) so the buffer
        is restored exactly. Restoration is in a ``finally`` so the live sampler
        state is never left perturbed."""
        if not self.debug:
            return
        xp = self.xp
        eng = buffer_obj._likelihood_engine
        params_phys = self.transform_fn.both_transforms(params_add, xp=cp)
        di = data_index.astype(xp.int32)
        eng.fill_template(buffer_obj.acs_buffer, params_phys, di, swap_N_vals,
                          factor=-1, waveform_kwargs=self.waveform_kwargs)
        try:
            buffer_obj.get_ll(params_add, data_index, data_index, swap_N_vals)
            d_h2 = xp.asarray(buffer_obj.d_h_out).real
            # residual r' = r - h_add  =>  <r'|h_add> = d_h_a - h_h_a
            expected = (xp.asarray(d_h_a) - xp.asarray(h_h_a)).real
            finite = xp.isfinite(d_h2) & xp.isfinite(expected)
            if bool(xp.any(finite)):
                diff = xp.abs(d_h2[finite] - expected[finite])
                scale = xp.maximum(xp.abs(expected[finite]), 1.0)
                relmax = float(xp.max(diff / scale))
                logger.info(
                    "[GB_DEBUG %s] residual add/remove round-trip: max rel diff = %.3e",
                    self.name, relmax,
                )
        finally:
            eng.fill_template(buffer_obj.acs_buffer, params_phys, di, swap_N_vals,
                              factor=+1, waveform_kwargs=self.waveform_kwargs)

    def _debug_seq_select(self, buffer_obj, band_sorter, ids, t_i, w_i, b_i,
                          slots, curr):
        """Pick the entry of this repeat batch to trace with the 3x3
        sequence figures: the chosen (walker, band) cell at its coldest
        temperature present, once per sampler step. With
        ``debug_seq_pick="loudest"`` (or a target f0 in mHz) the trace
        WAITS for the pick round that selects the cell's max-amplitude
        (or nearest-f0) source -- every source is picked exactly once per
        pass, so its round always comes. Returns None when the cell is
        absent, tracing is off, it already ran this step, or the picked
        source is not the requested one yet."""
        # WDM-tile-only (the 3x3 slab figures); no-op on the FD basis.
        if not self._debug_tiles or getattr(self, "_dbg_seq_done", True):
            return None
        try:
            sel_w = self.debug_plot_walker
            sel_b = (self.debug_plot_band if self.debug_plot_band is not None
                     else (len(self.band_edges) - 1) // 2)
            w_np = _to_numpy(w_i); b_np = _to_numpy(b_i); t_np = _to_numpy(t_i)
            match = np.where((w_np == sel_w) & (b_np == sel_b))[0]
            if match.size == 0:
                return None
            idx = int(match[np.argmin(t_np[match])])

            if self.debug_seq_pick != "first":
                # Target source of the traced cell: max amplitude
                # ("loudest") or nearest to an explicit f0 [mHz].
                cell_mask = (
                    (band_sorter.temp_inds == int(t_np[idx]))
                    & (band_sorter.walker_inds == sel_w)
                    & (band_sorter.band_inds == sel_b)
                    & band_sorter.inds
                )
                cell_ids = _to_numpy(
                    self.xp.arange(band_sorter.num_sources)[cell_mask])
                if cell_ids.size == 0:
                    return None
                cell_coords = _to_numpy(band_sorter.coords)[cell_ids]
                if self.debug_seq_pick == "loudest":
                    # SNR proxy, not bare amplitude: an edge-on source can
                    # carry the biggest amplitude at a fraction of the
                    # SNR. amp * sqrt(((1+cos^2 i)/2)^2 + cos^2 i) uses the
                    # sampling coords directly (col 4 = cos_iota); sky/psi
                    # response factors are O(1). Slot 0 is lnA (amplitude
                    # basis) or distance (distance basis) -> amplitude via
                    # the run transform either way.
                    c2 = cell_coords[:, 4] ** 2
                    _amp = _to_numpy(self.transform_fn.both_transforms(
                        cell_coords))[:, 0]
                    snr_proxy = _amp * np.sqrt(
                        ((1.0 + c2) / 2.0) ** 2 + c2)
                    target_id = int(cell_ids[np.argmax(snr_proxy)])
                else:
                    f0_target = float(self.debug_seq_pick)
                    target_id = int(cell_ids[
                        np.argmin(np.abs(cell_coords[:, 1] - f0_target))])
                if int(_to_numpy(ids)[idx]) != target_id:
                    return None  # not this round: keep waiting

            self._dbg_seq_done = True
            f0_old = float(_to_numpy(
                self.transform_fn.both_transforms(curr[idx:idx + 1], xp=cp)[0, 1]))
            return dict(
                idx=idx,
                slot=int(_to_numpy(slots)[idx]),
                temp=int(t_np[idx]), walker=sel_w, band=sel_b,
                f0_old=f0_old, f0_new=f0_old,
                snaps={},
            )
        except Exception as e:
            logger.warning("[GB_DEBUG %s] seq select skipped: %r", self.name, e)
            return None

    def _debug_slab_snapshot(self, buffer_obj, slot):
        """Copy one cell's residual slab as ``(nchannels, Nf_band, Nt_active)``.

        ``band_buffer[slot]`` is a PER-BAND slab whose frequency extent is the
        band's local layer count (``band_N``), NOT the full-domain
        ``bs.Nf_active``. Search mode's per-band leaf caps make the two differ
        (e.g. a 7-layer band slab vs. the 137-layer full domain), so infer Nf
        from the slab size rather than assuming the global value -- otherwise
        ``reshape`` raises ``cannot reshape array of size N``. Returns None on
        failure (debug-only: never break the sampler), matching the sibling
        ``_debug_cell_total_template`` / ``_debug_walker_true_data`` helpers.
        """
        try:
            bs = self._basis_settings
            Nt_a = int(getattr(bs, "Nt_active", None) or bs.Nt)
            nc = int(buffer_obj.nchannels)
            arr = _to_numpy(buffer_obj.band_buffer[slot]).copy()
            denom = nc * Nt_a
            if denom <= 0 or arr.size % denom != 0:
                logger.warning(
                    "[GB_DEBUG %s] slab snapshot skipped: size %d not "
                    "divisible by nc*Nt=%d", self.name, arr.size, denom,
                )
                return None
            Nf_band = arr.size // denom
            return arr.reshape(nc, Nf_band, Nt_a)
        except Exception as e:  # debug-only: never break the sampler
            logger.warning(
                "[GB_DEBUG %s] slab snapshot skipped: %r", self.name, e,
            )
            return None

    def _debug_cell_total_template(self, buffer_obj, band_sorter, seq):
        """Sum of ALL modeled templates of the traced cell (scratch fill).

        Fills every alive source of the traced (temp, walker, band) cell
        into a zeroed scratch slab through the engine's fill_template with
        factor=+1 -- the same sign convention as
        ``remove_sources_from_band_buffer`` (band_buffer = residual =
        data - sum(templates)), so ``residual + total_template`` is the
        cell's TOTAL DATA view. Returns None on failure (debug-only)."""
        try:
            t, w, b = seq["temp"], seq["walker"], seq["band"]
            mask = (
                (band_sorter.temp_inds == t)
                & (band_sorter.walker_inds == w)
                & (band_sorter.band_inds == b)
                & band_sorter.inds
            )
            bs = self._basis_settings
            Nt_a = int(getattr(bs, "Nt_active", None) or bs.Nt)
            nc = int(buffer_obj.nchannels)
            # PER-BAND slab geometry: match the traced cell's band_buffer slot
            # (search mode's per-band caps make Nf band-local, not the global
            # bs.Nf_active), so the scratch total-template combines cleanly with
            # the per-band residual snapshot (``before_removal + total``).
            _denom = nc * Nt_a
            _slab_sz = int(_to_numpy(buffer_obj.band_buffer[seq["slot"]]).size)
            Nf_a = (
                _slab_sz // _denom
                if _denom and _slab_sz % _denom == 0
                else int(getattr(bs, "Nf_active", None) or bs.Nf)
            )
            n_src = int(mask.sum())
            if n_src == 0:
                return np.zeros((nc, Nf_a, Nt_a))
            coords = band_sorter.coords[mask]
            params_phys = self.transform_fn.both_transforms(
                coords, xp=cp,
                leaf_inds=band_sorter.leaf_inds[mask] if self._per_leaf_fill else None,
            )
            scratch = cp.zeros(nc * Nf_a * Nt_a)

            class _Scratch:
                linear_data_arr = [scratch]

                def __len__(self):
                    return 1

            buffer_obj._likelihood_engine.fill_template(
                _Scratch(), params_phys,
                cp.zeros(n_src, dtype=cp.int32),
                band_sorter.band_N_vals[
                    cp.full(n_src, b, dtype=int)
                ],
                factor=+1, waveform_kwargs=self.waveform_kwargs,
            )
            return _to_numpy(scratch).reshape(nc, Nf_a, Nt_a)
        except Exception as e:  # debug-only: never break the sampler
            logger.warning(
                "[GB_DEBUG %s] cell total-template fill skipped: %r",
                self.name, e,
            )
            return None

    def _debug_walker_true_data(self, acs, walker):
        """The traced walker's TRUE data slab: injection data minus non-GB
        models, from the move's block-start snapshot
        (``reset_non_gb_linear_data_arr`` is taken with ALL cold-chain GB
        templates restored). Unlike the ``residual + cell templates``
        reconstruction, this is correct even when GB sources are modeled
        OUTSIDE the traced band (those subtractions are not undone by the
        cell's own templates). Returns None on failure (debug-only)."""
        try:
            snap = getattr(self, "reset_non_gb_linear_data_arr", None)
            if snap is None:
                return None
            bs = self._basis_settings
            Nf_a = int(getattr(bs, "Nf_active", None) or bs.Nf)
            Nt_a = int(getattr(bs, "Nt_active", None) or bs.Nt)
            nc = int(acs.nchannels)
            for i, split in enumerate(acs.gpu_splits):
                loc = np.where(np.asarray(split) == int(walker))[0]
                if loc.size:
                    arr = _to_numpy(snap[i]).reshape(-1, nc, Nf_a, Nt_a)
                    return arr[int(loc[0])].copy()
            return None
        except Exception as e:  # debug-only: never break the sampler
            logger.warning(
                "[GB_DEBUG %s] true-data slice skipped: %r", self.name, e,
            )
            return None

    def _debug_band_source_only_ll(self, buffer_obj, arr, slot, band):
        """Source-only ll ``-1/2 <a|a>`` of ``arr`` (nc, Nf_a, Nt_a), sliced
        to the WDM layers whose centers lie in ``band``'s edge interval
        (same slicing as :meth:`_debug_log_band_null`)."""
        bs = self._basis_settings
        layer_df = float(bs.layer_df)
        ind_min_f = int(bs.ind_min_f)
        Nf_a = arr.shape[1]
        if band is None:
            # full slab (all active layers) -- matches the kernel get_ll
            # support up to its per-source layer gating.
            k0, k1 = 0, Nf_a
        else:
            be = _to_numpy(self.band_edges)
            k0 = max(int(np.ceil(be[band] / layer_df - 1e-9)) - ind_min_f, 0)
            k1 = min(int(np.floor(be[band + 1] / layer_df + 1e-9)) + 1 - ind_min_f,
                     Nf_a)
        dc = float(buffer_obj.settings.differential_component)
        nc = buffer_obj.nchannels
        r = arr[:, k0:k1]
        psd_np = _to_numpy(buffer_obj._materialize(buffer_obj.psd_buffer)[slot])
        if buffer_obj.tdi_channel_setup == "XYZ":
            ic = psd_np.reshape(nc, nc, Nf_a, -1)[:, :, k0:k1]
            return -0.5 * 4.0 * dc * float(
                np.einsum("ifk,ijfk,jfk->", r, ic.real, r))
        ic = psd_np.reshape(nc, Nf_a, -1)[:, k0:k1]
        return -0.5 * 4.0 * dc * float(np.sum(r * ic.real * r))

    def _debug_plot_band_sequence(self, buffer_obj, seq) -> None:
        """Four 3x3 figures (rows = X/Y/Z channels; columns =
        |TOTAL template| / |TOTAL data| / |buffer residual|) at the four
        buffer moments of one in-model repeat block on the traced source:

            1 before_removal   (source modeled: residual ~ null)
            2 after_removal    (source UN-modeled: signal IN the residual)
            3 before_addback   (must equal 2 -- repeats never touch the buffer)
            4 after_addback    (final template re-subtracted: signal OUT)

        Column semantics (all reconstructed from the cell's constant TOTAL
        DATA, ``data_const = residual(before_removal) + sum of ALL modeled
        templates``, computed by ``_debug_cell_total_template``):

        - column 1 = ``data_const - residual`` = the SUM of the templates
          currently in the model: removing the picked source appears as a
          small DENT here (7 sources -> 6) ...
        - column 2 = ``data_const`` -- the total data view, identical in
          all four figures (every source visible);
        - column 3 = the RAW buffer residual -- ... and as +1 signal here.

        Each figure's suptitle carries the band's SOURCE-ONLY ll of the
        residual state (-1/2 <r|r> over the band's layers), so the in/out
        shows up numerically too. Buffer sign convention: band_buffer =
        RESIDUAL (data - templates); ``remove_sources_from_band_buffer``
        UN-models (residual += template), ``add_sources_...`` re-subtracts.
        """
        if not self.debug:
            return
        try:
            import os as _os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            s = seq["snaps"]
            need = ("before_removal", "after_removal",
                    "before_addback", "after_addback")
            if any(k not in s for k in need):
                return
            bs = self._basis_settings
            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)

            data_const = seq.get("data_const")
            t_tot = seq.get("t_tot")
            if data_const is None or t_tot is None:
                # Total-template fill failed at arm time: fall back to the
                # traced-source-only template diffs for column 1 and the
                # with-source pair state for column 2.
                data_const = s["after_removal"]
                t_tot = s["after_removal"] - s["before_removal"]
            lls = {k: self._debug_band_source_only_ll(
                       buffer_obj, s[k], seq["slot"], seq["band"])
                   for k in need}

            # Cross-check: the FULL-SLAB source-only delta-ll of the addback
            # (ll(after) - ll(before) = <r|h> - 1/2<h|h> analytically) must
            # match the LAST get_ll value of the repeat block (ll_ref of the
            # final accepted coordinates) up to the kernel's per-source
            # layer gating.
            ll_ref_final = seq.get("ll_ref_final")
            dll_addback = (
                self._debug_band_source_only_ll(
                    buffer_obj, s["after_addback"], seq["slot"], None)
                - self._debug_band_source_only_ll(
                    buffer_obj, s["before_addback"], seq["slot"], None)
            )
            if ll_ref_final is not None:
                logger.info(
                    "[GB_DEBUG %s] addback delta-ll (full slab) = %.6e vs "
                    "final get_ll = %.6e (diff %.3e)",
                    self.name, dll_addback, ll_ref_final,
                    abs(dll_addback - ll_ref_final),
                )
            # Total template at each moment: the block-start fill of ALL
            # the cell's modeled templates, minus what has been shifted
            # into the residual since (snap_k - snap_1 = the removed
            # template content). Built from the DIRECT fill -- not
            # data - residual -- so out-of-band models never leak into
            # this column when data_const is the true-data slab.
            def _t_at(k):
                return t_tot - (s[k] - s["before_removal"])

            figures = [
                # (tag, TOTAL template, TOTAL data, ACTUAL buffer state, f0)
                ("1_before_removal", _t_at("before_removal"),
                 data_const, s["before_removal"], seq["f0_old"]),
                ("2_after_removal", _t_at("after_removal"),
                 data_const, s["after_removal"], seq["f0_old"]),
                ("3_before_addback", _t_at("before_addback"),
                 data_const, s["before_addback"], seq["f0_new"]),
                ("4_after_addback", _t_at("after_addback"),
                 data_const, s["after_addback"], seq["f0_new"]),
            ]

            nc = data_const.shape[0]
            ch_names = ["X", "Y", "Z"][:nc]
            local = int(round(seq["f0_old"] / layer_df)) - ind_min_f
            # 5-layer (mm5-style) span: tight enough that neighboring
            # galaxy sources outside the band (e.g. 19.668 mHz, 5 layers
            # below the 20.38 mHz source) stay out of the figures.
            lo = max(local - 2, 0)
            hi = min(local + 3, data_const.shape[1])
            ylo = (ind_min_f + lo - 0.5) * layer_df * 1e3
            yhi = (ind_min_f + hi - 0.5) * layer_df * 1e3

            _os.makedirs(self.debug_plot_dir, exist_ok=True)
            # ONE color scale per channel row, shared by every column of
            # every figure in the sequence: the panels are directly
            # comparable, so the signal entering/leaving the buffer shows
            # up as brightness (a near-null residual stays dark instead of
            # being autoscaled up to look like signal).
            vmax_row = [
                max(float(np.abs(arr[row, lo:hi]).max())
                    for _tag, _T, _D, _R, _f0 in figures
                    for arr in (_T, _D, _R))
                for row in range(nc)
            ]
            for tag, T, D, R, f0 in figures:
                ll_state = lls[tag[2:]]
                fig, axes = plt.subplots(
                    nc, 3, figsize=(13.5, 3.2 * nc), squeeze=False,
                    sharex=True, sharey=True,
                )
                for row in range(nc):
                    for col, (name, arr) in enumerate(
                            [("total template", T), ("total data", D),
                             ("buffer residual", R)]):
                        ax = axes[row][col]
                        im = ax.imshow(
                            np.abs(arr[row, lo:hi]), aspect="auto",
                            origin="lower",
                            extent=[0, arr.shape[2], ylo, yhi],
                            vmin=0.0, vmax=vmax_row[row],
                        )
                        ax.axhline(f0 * 1e3, color="r", ls="--", lw=1.0)
                        if row == 0:
                            ax.set_title(f"|{name}|", fontsize=11)
                        if col == 0:
                            ax.set_ylabel(f"{ch_names[row]}\nfrequency [mHz]",
                                          fontsize=10)
                        if row == nc - 1:
                            ax.set_xlabel("WDM time pixel", fontsize=10)
                        ax.tick_params(labelsize=8)
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=7)
                extra = ""
                if tag.startswith("4_") and ll_ref_final is not None:
                    extra = (f"  |  addback $\\Delta$ll = {dll_addback:.4e} "
                             f"vs final get_ll = {ll_ref_final:.4e}")
                fig.suptitle(
                    f"GB in-model sequence {tag.replace('_', ' ')} — "
                    f"band {seq['band']} | walker {seq['walker']} | "
                    f"T{seq['temp']} | f0 = {f0 * 1e3:.4f} mHz\n"
                    f"band SOURCE-ONLY ll of buffer residual = "
                    f"{ll_state:.4e}{extra}",
                    fontsize=13,
                )
                fname = _os.path.join(
                    self.debug_plot_dir,
                    f"gb_debug_seq{tag}_band{seq['band']}_w{seq['walker']}"
                    f"_t{seq['temp']}_{self._dbg_plot_counter:04d}.png",
                )
                fig.savefig(fname, dpi=120, bbox_inches="tight")
                plt.close(fig)
                self._dbg_plot_counter += 1
                logger.info("[GB_DEBUG %s] saved sequence plot -> %s",
                            self.name, fname)
        except Exception as e:
            logger.warning("[GB_DEBUG %s] sequence plots skipped: %r",
                           self.name, e)

    def _debug_rj_select(self, buffer_obj, picked):
        """Arm the RJ before/after trace for the chosen (walker, band) cell
        (coldest temperature present in this pick round), once per step."""
        self._dbg_rj_seq = None
        # WDM-tile-only (before/after slab snapshots); no-op on the FD basis.
        if not self._debug_tiles or getattr(self, "_dbg_rj_done", True):
            return None
        try:
            sel_w = self.debug_plot_walker
            sel_b = (self.debug_plot_band if self.debug_plot_band is not None
                     else (len(self.band_edges) - 1) // 2)
            w_np = _to_numpy(picked["walker_inds"])
            b_np = _to_numpy(picked["band_inds"])
            t_np = _to_numpy(picked["temp_inds"])
            match = np.where((w_np == sel_w) & (b_np == sel_b))[0]
            if match.size == 0:
                return None
            idx = int(match[np.argmin(t_np[match])])
            slot = int(_to_numpy(picked["slot_index"])[idx])
            rj_seq = dict(
                idx=idx, slot=slot, temp=int(t_np[idx]),
                walker=sel_w, band=sel_b,
                before=self._debug_slab_snapshot(buffer_obj, slot),
            )
            # _run_rj_step marks rj_seq["accepted"] from the real accept
            # bookkeeping. A slab diff alone cannot be the signal: the
            # verify hook's add/remove round-trips leave ~1e-10-relative
            # FP dust in the slab even for rejected proposals.
            self._dbg_rj_seq = rj_seq
            return rj_seq
        except Exception as e:
            logger.warning("[GB_DEBUG %s] rj select skipped: %r", self.name, e)
            return None

    def _debug_plot_rj_pair(self, buffer_obj, rj_seq) -> None:
        """After the RJ step: if the traced cell's RJ proposal was ACCEPTED
        (per the accept bookkeeping recorded by ``_run_rj_step``), save ONE
        3x3 figure -- rows = channels, columns = |accepted template|
        (after - before) / |buffer before RJ| / |buffer after RJ| -- with
        the band's source-only ll of both states in the title. No figure
        when the proposal was rejected; the slab may still differ by FP
        dust from the verify hook's round-trips, which is why the accept
        flag (not the diff) is the gate."""
        if rj_seq is None or not self.debug:
            return
        try:
            import os as _os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            after = self._debug_slab_snapshot(buffer_obj, rj_seq["slot"])
            before = rj_seq["before"]
            diff = after - before
            self._dbg_rj_seq = None
            if not rj_seq.get("accepted", False):
                return  # traced cell's RJ proposal was rejected
            self._dbg_rj_done = True

            bs = self._basis_settings
            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)
            ll_b = self._debug_band_source_only_ll(
                buffer_obj, before, rj_seq["slot"], rj_seq["band"])
            ll_a = self._debug_band_source_only_ll(
                buffer_obj, after, rj_seq["slot"], rj_seq["band"])

            # Center the view on the accepted template's peak layer.
            prof = np.abs(diff).sum(axis=(0, 2))
            local = int(np.argmax(prof))
            lo = max(local - 2, 0)
            hi = min(local + 3, diff.shape[1])
            ylo = (ind_min_f + lo - 0.5) * layer_df * 1e3
            yhi = (ind_min_f + hi - 0.5) * layer_df * 1e3

            nc = diff.shape[0]
            ch_names = ["X", "Y", "Z"][:nc]
            vmax_row = [max(float(np.abs(a[row, lo:hi]).max())
                            for a in (diff, before, after))
                        for row in range(nc)]
            fig, axes = plt.subplots(
                nc, 3, figsize=(13.5, 3.2 * nc), squeeze=False,
                sharex=True, sharey=True,
            )
            for row in range(nc):
                for col, (name, arr) in enumerate(
                        [("accepted template", diff),
                         ("buffer before RJ", before),
                         ("buffer after RJ", after)]):
                    ax = axes[row][col]
                    im = ax.imshow(
                        np.abs(arr[row, lo:hi]), aspect="auto", origin="lower",
                        extent=[0, arr.shape[2], ylo, yhi],
                        vmin=0.0, vmax=vmax_row[row],
                    )
                    if row == 0:
                        ax.set_title(f"|{name}|", fontsize=11)
                    if col == 0:
                        ax.set_ylabel(f"{ch_names[row]}\nfrequency [mHz]",
                                      fontsize=10)
                    if row == nc - 1:
                        ax.set_xlabel("WDM time pixel", fontsize=10)
                    ax.tick_params(labelsize=8)
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.ax.tick_params(labelsize=7)
            fig.suptitle(
                f"GB rj ACCEPTED — band {rj_seq['band']} | "
                f"walker {rj_seq['walker']} | T{rj_seq['temp']}\n"
                f"band SOURCE-ONLY ll: before = {ll_b:.4e}, "
                f"after = {ll_a:.4e} (Δ = {ll_a - ll_b:+.4e})",
                fontsize=13,
            )
            _os.makedirs(self.debug_plot_dir, exist_ok=True)
            fname = _os.path.join(
                self.debug_plot_dir,
                f"gb_debug_seq0_rj_accepted_band{rj_seq['band']}"
                f"_w{rj_seq['walker']}_t{rj_seq['temp']}"
                f"_{self._dbg_plot_counter:04d}.png",
            )
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            self._dbg_plot_counter += 1
            logger.info("[GB_DEBUG %s] saved RJ before/after plot -> %s",
                        self.name, fname)
        except Exception as e:
            logger.warning("[GB_DEBUG %s] rj pair plot skipped: %r",
                           self.name, e)

    def _debug_log_band_null(self, buffer_obj) -> None:
        """Log the CHOSEN band's source-only residual log-likelihood per
        temperature, once per sampler step (right after the first buffer
        load, i.e. central-band sources subtracted, before any proposal).

        This is the direct null check: with noiseless data and the cold
        chain at (or near) the injection, the chosen band's
        ``-1/2 <r|r>`` restricted to the band's WDM layers must be ~0 for
        T0. The full-buffer ``likelihood(source_only=True)`` cannot show
        this on the WDM path -- each cell slab spans the whole active band,
        so the unsubtracted rest of the galaxy dominates; this slices out
        exactly the band's layers.
        """
        # WDM-tile-only (slices the slab by WDM layers); no-op on the FD basis.
        if not self._debug_tiles or getattr(self, "_dbg_null_logged", True):
            return
        try:
            bs = self._basis_settings
            if not hasattr(bs, "layer_df"):
                return  # WDM-only diagnostic for now
            sel_w = self.debug_plot_walker
            sel_b = (self.debug_plot_band if self.debug_plot_band is not None
                     else (len(self.band_edges) - 1) // 2)

            combos = _to_numpy(buffer_obj.unique_band_combos)
            rows = [i for i in range(combos.shape[0])
                    if int(combos[i, 1]) == sel_w and int(combos[i, 2]) == sel_b]
            if not rows:
                return
            self._dbg_null_logged = True

            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)
            Nf_a = int(getattr(bs, "Nf_active", None) or bs.Nf)
            Nt_a = int(getattr(bs, "Nt_active", None) or bs.Nt)
            be = _to_numpy(self.band_edges)
            # WDM layers are CENTERED on m*layer_df while band edges are at
            # m*layer_df, so the band interval straddles two layers: include
            # every layer whose center lies in [edge_b, edge_b+1] (both
            # boundary layers), so the carrier layer is always covered.
            k0 = max(int(np.ceil(be[sel_b] / layer_df - 1e-9)) - ind_min_f, 0)
            k1 = min(int(np.floor(be[sel_b + 1] / layer_df + 1e-9)) + 1 - ind_min_f,
                     Nf_a)
            dc = float(buffer_obj.settings.differential_component)
            nc = buffer_obj.nchannels

            band_np = _to_numpy(buffer_obj._materialize(buffer_obj.band_buffer))
            psd_np = _to_numpy(buffer_obj._materialize(buffer_obj.psd_buffer))
            msgs = []
            for i in sorted(rows, key=lambda i: int(combos[i, 0])):
                t = int(combos[i, 0])
                r = band_np[i].reshape(nc, Nf_a, Nt_a)[:, k0:k1]
                if buffer_obj.tdi_channel_setup == "XYZ":
                    ic = psd_np[i].reshape(nc, nc, Nf_a, Nt_a)[:, :, k0:k1]
                    ll = -0.5 * 4.0 * dc * float(
                        np.einsum("ifk,ijfk,jfk->", r, ic.real, r))
                else:
                    ic = psd_np[i].reshape(nc, Nf_a, Nt_a)[:, k0:k1]
                    ll = -0.5 * 4.0 * dc * float(np.sum(r * ic.real * r))
                msgs.append(f"T{t}: {ll:.6e}")
            logger.info(
                "[GB_DEBUG %s] sub-band SOURCE-ONLY residual ll "
                "(band %d, walker %d, layers %d:%d): %s "
                "(cold chain at injection should be ~0)",
                self.name, sel_b, sel_w, ind_min_f + k0, ind_min_f + k1,
                "; ".join(msgs),
            )
        except Exception as e:
            logger.warning("[GB_DEBUG %s] band-null log skipped: %r", self.name, e)

    def _debug_plot_band(self, buffer_obj, params_add, data_index, swap_N_vals,
                         ll_diff_kept, map_to_update_cpu, keep2, move_i,
                         stage: str = "in-model") -> None:
        """Save ONE WDM time-frequency figure for the CHOSEN (walker, band)
        cell, with a panel per temperature present in this proposal batch.

        Only the cell selected by ``debug_plot_walker`` / ``debug_plot_band``
        (default: walker 0 / the central band) is plotted -- not every
        (temp, walker, band) the proposal touches -- so a debug run produces
        a small, readable progression instead of hundreds of PNGs.

        ``stage`` labels the proposal context in the title and file name:
        ``"rj"`` (birth/death step) or ``"in-model"`` (repeat block).
        """
        # WDM time-frequency tile figure; no-op on the FD basis.
        if not self._debug_tiles:
            return
        try:
            import os as _os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xp = self.xp
            bs = self._basis_settings
            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)

            sel_w = self.debug_plot_walker
            sel_b = (self.debug_plot_band if self.debug_plot_band is not None
                     else (len(self.band_edges) - 1) // 2)

            # ``params_add`` / ``data_index`` / ``ll_diff_kept`` are aligned
            # with the KEPT subset; map_to_update_cpu indexes the full batch,
            # bridged by ``orig``.
            orig = np.where(_to_numpy(keep2).astype(bool))[0]
            if orig.size == 0:
                return
            temps_all, walkers_all, bands_all = map_to_update_cpu
            pos = [i for i, j in enumerate(orig)
                   if int(walkers_all[j]) == sel_w and int(bands_all[j]) == sel_b]
            if not pos:
                return  # chosen cell not in this batch
            # One figure per stage per sampler step: the set is reset at the
            # top of run_proposal, so later picks of the same cell within
            # this step do not save additional figures.
            plotted = getattr(self, "_dbg_plotted_stages", None)
            if plotted is not None:
                if stage in plotted:
                    return
                plotted.add(stage)
            pos.sort(key=lambda i: int(temps_all[orig[i]]))

            params_phys = self.transform_fn.both_transforms(params_add, xp=cp)
            di_np = _to_numpy(data_index)
            ll_np = _to_numpy(xp.asarray(ll_diff_kept))

            # band_buffer is per-slot (num_bands_now, nchannels, data_length)
            # with the WDM tile flattened; recover (Nf_active, Nt_active).
            Nf_a = int(getattr(bs, "Nf_active", None) or bs.Nf)
            Nt_a = int(getattr(bs, "Nt_active", None) or bs.Nt)

            n = len(pos)
            ncols = min(n, 4)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5.6 * ncols, 4.4 * nrows),
                squeeze=False, sharey=True,
            )
            for ax in axes.flat[n:]:
                ax.set_visible(False)

            for panel, i0 in enumerate(pos):
                ax = axes.flat[panel]
                temp = int(temps_all[orig[i0]])
                slab = int(di_np[i0])
                f0 = float(_to_numpy(params_phys[i0, 1]))
                local = int(round(f0 / layer_df)) - ind_min_f
                ll_val = float(ll_np[i0])
                tile = np.abs(
                    _to_numpy(buffer_obj.band_buffer[slab][0])
                ).reshape(Nf_a, Nt_a)
                # 5-layer (mm5-style) span, matching the sequence figures.
                lo = max(local - 2, 0)
                hi = min(local + 3, tile.shape[0])
                sub = tile[lo:hi]
                # WDM layer m is CENTERED on m*layer_df (span (m +- 1/2)*df):
                # the y-extent runs from the bottom edge of layer lo to the
                # top edge of layer hi-1, i.e. offset -1/2 layer relative to
                # the raw indices. (Without the -0.5 every row displayed a
                # half-layer too high and sources looked misaligned.)
                im = ax.imshow(
                    sub, aspect="auto", origin="lower",
                    extent=[0, sub.shape[1],
                            (ind_min_f + lo - 0.5) * layer_df * 1e3,
                            (ind_min_f + hi - 0.5) * layer_df * 1e3],
                )
                ax.axhline(f0 * 1e3, color="r", ls="--", lw=1.2,
                           label=f"f0 = {f0 * 1e3:.4f} mHz")
                # RJ forbidden proposals carry a -1e300 sentinel, not a
                # likelihood -- label them instead of printing the sentinel.
                ll_txt = ("forbidden proposal" if ll_val < -1e290
                          else f"$\\Delta$logL = {ll_val:.3e}")
                ax.set_title(f"T{temp}  {ll_txt}", fontsize=11)
                ax.set_xlabel("WDM time pixel (X)", fontsize=10)
                if panel % ncols == 0:
                    ax.set_ylabel("frequency [mHz]", fontsize=10)
                ax.tick_params(labelsize=9)
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
                cbar = fig.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=8)

            fig.suptitle(
                f"GB {stage} proposal — |WDM residual| around the source | "
                f"band {sel_b} | walker {sel_w} | repeat {move_i} | "
                f"all temperatures",
                fontsize=13,
            )
            _os.makedirs(self.debug_plot_dir, exist_ok=True)
            fname = _os.path.join(
                self.debug_plot_dir,
                f"gb_debug_{stage.replace('-', '')}_band{sel_b}_w{sel_w}"
                f"_move{move_i}_{self._dbg_plot_counter:04d}.png",
            )
            # bbox_inches="tight" guarantees the suptitle, axis labels, and
            # colorbar labels are all inside the saved figure.
            fig.savefig(fname, dpi=130, bbox_inches="tight")
            plt.close(fig)
            self._dbg_plot_counter += 1
            logger.info("[GB_DEBUG %s] saved band plot -> %s", self.name, fname)
        except Exception as e:
            logger.warning("[GB_DEBUG %s] band plot skipped: %r", self.name, e)

    def _fstat_reference_walker(self, model):
        """Max-likelihood walker used as the F-stat distance-birth reference.

        Mirrors the serial-search move: the F-stat centers are computed
        against the residual of the best-fitting walker. Computed once per
        ``run_proposal`` (the residual drifts within a proposal, but the
        reference only sets the proposal CENTER, not the accept test).

        TODO (fstat refit cadence): the F-stat PEAK GRID (band_peaks_stacked
        .npz driving the birth container's intrinsics) is built ONCE up front
        against the initial residual, so it goes stale as sources are
        subtracted. Examine refitting the F-stat (grid + this reference) at the
        BEGINNING OF EACH PROPOSAL so births track the evolving residual --
        gated on the wall-time cost of the refit (grid rebuild was ~7s comb +
        ~13s stage-B; per-proposal that may or may not pay for itself vs. the
        ~130s/iter dominated by in-model repeats). Measure before adopting.
        """
        try:
            return int(np.argmax(_to_numpy(model.analysis_container_arr.likelihood())))
        except Exception:
            return 0

    def run_proposal(self, model, state, band_sorter, band_temps):
        """One full pass of per-band proposals.

        For each band-parity unit: *open* the parity class (cold-chain
        templates restored into the parent residual so every central-band
        window holds coordinate-independent raw data), load the
        (temp, walker, band) cells into the sub-band buffer, then repeatedly
        pick one not-yet-visited source per active cell (random order,
        without replacement) and run the RJ step plus the in-model repeats
        on it. Finished cells are swapped out for pending ones. *Closing*
        the parity class re-subtracts the cold-chain templates with the
        possibly-updated coordinates -- that is what propagates accepted
        cold-chain changes into the parent residual for the next unit and
        for the tempering stage.

        Returns ``(ll_change_log, prop_counts, acc_counts)``; the count
        arrays have shape ``(2, ntemps, nwalkers, num_bands)`` with row 0 =
        RJ proposals and row 1 = in-model proposals.
        """
        ll_change_log = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))
        prop_counts = cp.zeros((2, self.ntemps, self.nwalkers, self.num_bands), dtype=int)
        acc_counts = cp.zeros_like(prop_counts)

        # One debug figure per STAGE per run_proposal call (i.e. per sampler
        # step): _debug_plot_band consumes this set. The band-null log fires
        # once per step too.
        self._dbg_plotted_stages = set()
        self._dbg_null_logged = False
        self._dbg_seq_done = False
        self._dbg_rj_done = False

        # Reference walker for the F-stat distance-birth proposal center
        # (computed once per proposal; see _fstat_reference_walker).
        self._fstat_walker_ref = (
            self._fstat_reference_walker(model)
            if self.rj_fstat_dist_birth else 0
        )

        units = self.band_units if self.num_bands > 1 else 1
        start_unit = model.random.randint(units)

        for unit_i in range(units):
            remainder = (start_unit + unit_i) % units

            if self.debug:
                _dbg_ll_unit_start = _to_numpy(
                    model.analysis_container_arr.likelihood()
                ).copy()
                _dbg_change_start = _to_numpy(ll_change_log[0].sum(axis=-1)).copy()

            # Open this parity class in the parent residual.
            with _tspan(getattr(self, "_prop_timer", None), "unit_open_close"):
                self.remove_cold_chain_sources_from_residual(
                    model, band_sorter, units=units, remainder=remainder
                )
            self._debug_cold_chain_residual_loaded(model, remainder)

            apply_inds = not self.is_rj_prop
            
            extra_bool = (
                (band_sorter.band_inds < self.num_bands - 1) & (band_sorter.band_inds > 0)
            ) if self.num_bands > 1 else None

            subset = band_sorter.get_subset(
                units=units,
                remainder=remainder,
                apply_inds=apply_inds,
                extra_bool=extra_bool,
            )
            if subset is not None:
                self._run_band_unit(
                    model, band_sorter, subset, band_temps,
                    ll_change_log, prop_counts, acc_counts,
                )

            # Close: re-subtract with (possibly updated) cold-chain coords.
            with _tspan(getattr(self, "_prop_timer", None), "unit_open_close"):
                self.add_cold_chain_sources_to_residual(
                    model, band_sorter, units=units, remainder=remainder
                )

            if self.debug:
                _dbg_ll_unit_end = _to_numpy(
                    model.analysis_container_arr.likelihood()
                )
                _dbg_change_end = _to_numpy(ll_change_log[0].sum(axis=-1))
                _direct = _dbg_ll_unit_end - _dbg_ll_unit_start
                _tracked = _dbg_change_end - _dbg_change_start
                logger.info(
                    "[GB_DEBUG %s] unit %d (remainder %d) parent-ll reconcile: "
                    "direct per-walker %s vs tracked %s (max abs diff %.3e)",
                    self.name, unit_i, remainder,
                    np.array2string(_direct, precision=3),
                    np.array2string(_tracked, precision=3),
                    float(np.abs(_direct - _tracked).max()),
                )

            with _tspan(getattr(self, "_prop_timer", None), "mempool_free"):
                if self.backend.uses_cupy:
                    self.xp.cuda.runtime.deviceSynchronize()
                self.mempool.free_all_blocks()

        return ll_change_log, prop_counts, acc_counts

    def _cached_get_buffer(self, sorter, acs, specials, **kwargs):
        """Propose-scoped SubBandBuffer reuse (ONE cached ACA per signature).

        First call for a given ``(cell count, construction kwargs)``
        signature allocates through ``get_buffer``; every later same-
        signature call performs a FULL rebind of that cached ACA (the
        existing ``inds_fill``/``buffer_obj`` path — construction minus
        allocation), so the steady state is one allocation per signature
        per proposal instead of one per parity unit. The template twin
        rides in the signature: proposal-phase buffers never carry it (its
        guarded fill paths are tempering-only). The cache lives from
        ``propose`` start to ``return`` (see the lifecycle block there).
        """
        cache = getattr(self, "_prop_buffer_cache", None)
        if cache is None:
            cache = self._prop_buffer_cache = {}
        key = (int(specials.shape[0]), tuple(sorted(kwargs.items())))
        buf = cache.get(key)
        if buf is None:
            buf = sorter.get_buffer(acs, specials, **kwargs)
            cache[key] = buf
            self._prop_buffer_builds = getattr(self, "_prop_buffer_builds", 0) + 1
        else:
            sorter.get_buffer(
                acs, specials,
                inds_fill=self.xp.arange(int(specials.shape[0])),
                buffer_obj=buf,
            )
        return buf

    def _buffer_cache_teardown(self):
        """Drop the propose-scoped buffer cache and clear its device memory.

        Proposal-exit contract (memory-lifecycle rule): SubBandBuffer scratch
        is strictly proposal-scoped — when the proposal returns state, every
        shard's buffers are dropped and each owning device's pool is swept so
        the memory is available to the other modules' moves. Main-ACA / DCGA
        persistent allocations are untouched. Engine bindings and shard
        views cache on the buffers themselves, so they die here with the
        cache (a fresh proposal's buffers start with no stale bindings).
        """
        cache = getattr(self, "_prop_buffer_cache", None)
        n_builds = getattr(self, "_prop_buffer_builds", 0)
        devices = set()
        for buf in (cache or {}).values():
            for dev in (getattr(buf, "gpus", None) or []):
                devices.add(int(dev))
        self._prop_buffer_cache = None
        self._prop_buffer_builds = 0
        if not self.backend.uses_cupy:
            return
        used = self.mempool.used_bytes() / 1e9
        total = self.mempool.total_bytes() / 1e9
        logger.info(
            "%s: buffer lifecycle -- %d allocation(s) this propose "
            "(%d cached signature(s)); GPU pool used %.2f / total %.2f GB.",
            self.name, n_builds, len(cache or {}), used, total,
        )
        self.xp.cuda.runtime.deviceSynchronize()
        if not devices:
            # gpus=None CUDA mode: everything lives on the current device.
            self.mempool.free_all_blocks()
            return
        main_dev = self.xp.cuda.runtime.getDevice()
        try:
            for dev in sorted(devices):
                with self.xp.cuda.Device(dev):
                    self.xp.cuda.runtime.deviceSynchronize()
                    self.xp.get_default_memory_pool().free_all_blocks()
        finally:
            self.xp.cuda.runtime.setDevice(main_dev)

    def _run_band_unit(self, model, band_sorter, subset, band_temps,
                       ll_change_log, prop_counts, acc_counts):
        """Drive one parity unit's cells through the sub-band buffer."""
        tm = getattr(self, "_prop_timer", None)
        scheduler = BandScheduler(
            subset.special_band_inds, self.num_band_preload, xp=self.xp
        )
        with _tspan(tm, "buffer_build"):
            buffer_obj = self._cached_get_buffer(
                subset, model.analysis_container_arr,
                scheduler.slot_specials.copy(),
            )
        if tm is not None:
            tm.count("cells", int(scheduler.n_cells))
        self._debug_log_band_null(buffer_obj)

        # Pick eligibility lives on the MAIN sorter: only sources inside this
        # unit's subset are candidates (for in-model moves the subset already
        # applied ``inds``; for RJ it includes the freshly-drawn dead ones).
        eligible = self.xp.zeros(band_sorter.num_sources, dtype=bool)
        eligible[subset.inds_main_band_sorter] = True

        round_i = 0
        while scheduler.any_active():
            with _tspan(tm, "pick"):
                picked = self._pick_sources(band_sorter, buffer_obj, scheduler, eligible)
            if picked is None:
                break
            if tm is not None:
                # Batch size per repeat round: on GPU, small batches mean the
                # 100-repeat in-model loop is kernel-launch-overhead-bound.
                tm.count("picked_sources", int(len(picked["specials"])))

            if self.is_rj_prop:
                # RJ before/after trace of the chosen cell: snapshots
                # bracket the RJ step; figures save only when the cell's RJ
                # proposal was ACCEPTED (buffer changed). Chronologically
                # BEFORE the in-model sequence figures.
                rj_seq = self._debug_rj_select(buffer_obj, picked)
                with _tspan(tm, "rj_step"):
                    self._run_rj_step(
                        model, band_sorter, buffer_obj, band_temps, picked,
                        ll_change_log, prop_counts, acc_counts, round_i, scheduler,
                    )
                self._debug_plot_rj_pair(buffer_obj, rj_seq)

            with _tspan(tm, "inmodel_repeats"):
                self._run_in_model_repeats(
                    model, band_sorter, buffer_obj, band_temps, picked,
                    ll_change_log, prop_counts, acc_counts,
                )

            scheduler.record_picks(picked["specials"])
            inds_fill, new_specials = scheduler.advance()
            if len(inds_fill):
                with _tspan(tm, "buffer_build"):
                    subset.get_buffer(
                        model.analysis_container_arr, new_specials,
                        inds_fill=inds_fill, buffer_obj=buffer_obj,
                    )
                self._debug_log_band_null(buffer_obj)
            round_i += 1
            # GPU efficiency (parallel-resources plan P1): freeing the WHOLE
            # CuPy pool every pick round forces cudaFree/cudaMalloc churn
            # for every allocation in the next round, so it is now OPT-IN —
            # set GB_MEMPOOL_FREE_EACH_ROUND=1 only when a run is genuinely
            # memory-bound (the per-unit/per-proposal frees remain). The
            # mempool_free stage time quantifies the cost either way.
            if os.environ.get("GB_MEMPOOL_FREE_EACH_ROUND", "0") == "1":
                with _tspan(tm, "mempool_free"):
                    self.mempool.free_all_blocks()
        if tm is not None:
            tm.count("pick_rounds", round_i)

        logger.info(
            f"{self.name}: band unit complete after {round_i} pick rounds "
            f"({scheduler.n_cells} cells)."
        )

    def _pick_sources(self, band_sorter, buffer_obj, scheduler, eligible):
        """One not-yet-visited source per active cell, without replacement.

        Vectorized on ``self.xp``: candidates are gathered through the
        special-index maps, randomly ranked within each cell, and the first
        per cell wins. ``band_sorter.has_run_rj`` marks consumed sources for
        the remainder of this proposal, so every source is visited exactly
        once per pass.
        """
        xp = self.xp
        cand = (
            eligible
            & (~band_sorter.has_run_rj)
            & band_sorter.get_subset_bool(
                special_band_inds=scheduler.active_slot_specials
            )
        )
        cand_ids = xp.arange(band_sorter.num_sources)[cand]
        if len(cand_ids) == 0:
            return None

        # Random rank within each cell: specials are integers spaced >= 1,
        # so adding U[0, 0.5) keeps the cell blocks intact while randomizing
        # the within-cell order (robust to non-stable argsort).
        specials = band_sorter.special_band_inds[cand_ids]
        key = specials.astype(xp.float64) + xp.random.rand(len(specials)) * 0.5
        order = xp.argsort(key)
        ids_sorted = cand_ids[order]
        _, first = xp.unique(specials[order], return_index=True)
        ids = ids_sorted[first]

        band_sorter.has_run_rj[ids] = True

        specials_picked = band_sorter.special_band_inds[ids]
        band_inds = band_sorter.band_inds[ids]
        return {
            "ids": ids,
            "specials": specials_picked,
            "slot_index": buffer_obj.get_index(specials_picked).astype(xp.int32),
            "temp_inds": band_sorter.temp_inds[ids],
            "walker_inds": band_sorter.walker_inds[ids],
            "band_inds": band_inds,
            "N_vals": band_sorter.band_N_vals[band_inds].copy(),
        }

    def _log_dist_range(self, band_sorter):
        """``log(width)`` of the birth container's uniform distance (slot 0).

        The container draws ``dist ~ U[dist_lims]``, so its contribution to the
        proposal ``logpdf`` for slot 0 is the constant ``-log(width)``
        regardless of the drawn value. The F-stat distance proposal replaces
        that term, so we need ``log(width)`` to neutralize it in the RJ factor.
        Cached after first lookup.
        """
        if self._log_dist_range_cache is not None:
            return self._log_dist_range_cache
        cont = getattr(band_sorter, "rj_prop", None)
        if isinstance(cont, dict):
            cont = cont.get(self.branch_name)
        val = float(np.log(40.0 - 0.001))  # distance-basis default fallback
        try:
            for inds, dist in cont.priors:
                if 0 in list(inds) and hasattr(dist, "width"):
                    val = float(np.log(float(dist.width)))
                    break
        except Exception:
            pass
        self._log_dist_range_cache = val
        return val

    def _fstat_NM(self, model, params_phys, walker_ref):
        """Per-binary F-stat ``(N, M)`` from the domain-appropriate comp.

        Domain is the ONLY thing that differs between FD and WDM: pick the
        matching comp and call its ``get_fstat_ll_{fd,wdm}`` against the
        reference walker's residual (``model.analysis_container_arr``). Both
        return the same ``(N (num_bin,4), M_upper (num_bin,10))`` layout at the
        same fixed basis-filter reference, so everything downstream -- the
        Jaranowski-Krol maximization -- is shared and identical. NOTE: routed
        through the NEW GBFDComputations / GBWDMComputations, never the legacy
        SharedMemoryGBGPU ``gb.get_fstat_ll`` (FD-only; floods a WDM buffer).

        Multi-GPU: the comps are single-shard by contract, so the call goes
        through :meth:`_RoutedBandEngine.route_fstat_ll` (the raw-comp shard
        route, same fashion as ``get_ll`` / ``route_information_matrix``) --
        single-shard holders pass straight through; on a sharded parent ACA
        the reference walker's rows run on its owning device and (N, M) come
        back on the caller's device.
        """
        xp = self.xp
        di = xp.full(params_phys.shape[0], int(walker_ref), dtype=xp.int32)
        holder = model.analysis_container_arr
        comp_method = (
            self.gb_fd_comp.get_fstat_ll_fd
            if isinstance(self._basis_settings, FDSettings)
            else self.gb_wdm_comp.get_fstat_ll_wdm
        )
        return _RoutedBandEngine.route_fstat_ll(
            comp_method, holder, params_phys,
            data_index=di, noise_index=di, convert_to_ra_dec=False)

    def _fstat_dist_centers(self, model, rows_params, walker_ref):
        """F-stat 4-parameter extrinsic maxima for a set of birth/death rows.

        Transforms ``rows_params`` to physical, gets the F-stat ``(N, M)`` from
        the domain-appropriate comp (:meth:`_fstat_NM`), then runs the SINGLE
        shared Jaranowski-Krol inversion
        (:func:`lisatools.sampling.fstat_proposal.fstat_maximized_extrinsics`)
        -- identical for FD and WDM -- to recover ``(A_max, phi0_max, iota_max,
        psi_max, F)`` (``SNR^2 = 2F``). The amplitude/phase/iota/psi input
        columns are ignored by the F-stat, so the placeholder distance in
        ``rows_params`` does not matter. Returns arrays on ``self.xp``.
        """
        from ...sampling.fstat_proposal import fstat_maximized_extrinsics

        xp = self.xp
        # physical layout: [A, f0, fdot, fddot, phi0, iota, psi, alpha, delta]
        x_phys = self.transform_fn.both_transforms(rows_params, xp=xp)
        N_arr, M_upper = self._fstat_NM(model, x_phys, walker_ref)
        A_max, phi0_max, iota_max, psi_max, F = fstat_maximized_extrinsics(
            N_arr, M_upper)
        return (
            xp.asarray(A_max), xp.asarray(phi0_max),
            xp.asarray(iota_max), xp.asarray(psi_max), xp.asarray(F),
        )

    def _dist_center_and_width(self, rows_params, A_max, F):
        """``(ln_center, sigma)`` of the slot-0 log proposal from the F-stat.

        Distance basis: center ``ln dist* = ln(gb_amp_from_dist(f0,Mc,1)/A_max)``.
        Amplitude basis: center ``ln A* = ln A_max`` (slot 0 is lnA directly).
        Width ``sigma = 1/SNR`` with ``SNR = sqrt(max(2F, 1))`` (fractional
        amplitude/distance uncertainty from the F-stat curvature), floored so a
        weak/off-peak F-stat gives a broad -- not degenerate -- proposal.
        """
        xp = self.xp
        A_max = xp.clip(A_max, 1e-300, None)
        snr = xp.sqrt(xp.clip(2.0 * F, 1.0, None))
        sigma = 1.0 / snr
        if _gb_use_distance(self):
            from ..stock.erebor.transforms import gb_amp_from_dist
            k_amp = gb_amp_from_dist(rows_params[:, 1] * 1e-3, rows_params[:, 2], 1.0)
            ln_center = xp.log(xp.clip(k_amp / A_max, 1e-300, None))
        else:
            ln_center = xp.log(A_max)
        return ln_center, sigma

    def _slot0_log_proposal(self, slot0_vals, ln_center, sigma):
        """``log g`` of the slot-0 value under the F-stat lognormal proposal.

        The proposal is Gaussian in the LOG of slot 0 (log-distance or lnA),
        so the density in the sampled coordinate ``v`` is
        ``g(v) = N(ln v; ln_center, sigma) / v`` (the ``1/v`` is the
        ``ln v -> v`` Jacobian). For the amplitude basis slot 0 is ALREADY lnA
        (sampled in log space), so there is no Jacobian term there.
        """
        xp = self.xp
        lv = xp.log(xp.clip(slot0_vals, 1e-300, None)) if _gb_use_distance(self) else slot0_vals
        logg = (
            -0.5 * ((lv - ln_center) / sigma) ** 2
            - xp.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        )
        if _gb_use_distance(self):
            logg = logg - lv  # Jacobian d(ln dist)/d(dist) = 1/dist
        return logg

    def _run_rj_step(self, model, band_sorter, buffer_obj, band_temps, picked,
                     ll_change_log, prop_counts, acc_counts, round_i, scheduler):
        """Birth/death proposal for each picked source (vectorized over cells).

        Births (``inds == False``; coordinates pre-drawn from the RJ proposal
        distribution in the BandSorter) score the add delta
        ``<r|h> - 0.5<h|h>``; deaths (``inds == True``) score the removal
        delta ``-<r|h> - 0.5<h|h>`` (see
        :meth:`SubBandBuffer.get_removal_ll`). Detailed-balance factors are
        the pre-computed ±logpdf of the RJ proposal distribution. On accept,
        ``inds`` flips and the cell residual is updated through
        ``fill_template`` with the appropriate sign.
        """
        xp = self.xp
        ids = picked["ids"]
        slots = picked["slot_index"]
        N_vals = picked["N_vals"]
        alive = band_sorter.inds[ids].copy()   # True -> death proposal

        params = band_sorter.coords[ids].copy()
        params[:] = self.periodic.wrap({self.branch_name: params[:, None, :]}, xp=xp)[self.branch_name][:, 0]

        logp = cp.asarray(self.gpu_priors[self.branch_name].logpdf(params))
        prev_logp = cp.zeros_like(logp)
        curr_logp = cp.zeros_like(logp)
        prev_logp[alive] = logp[alive]
        curr_logp[~alive] = logp[~alive]

        # Births outside this cell's frequency window are unphysical.
        f_hz = params[:, 1] / 1e3
        out_of_band = (
            (f_hz < buffer_obj.frequency_lims[0][slots])
            | (f_hz > buffer_obj.frequency_lims[1][slots])
        )
        curr_logp[(~alive) & out_of_band] = -np.inf

        # Per-band progressive leaf cap (search mode): a birth into a band
        # already holding ``cap[b]`` alive sources is prior-forbidden --
        # a truncation of the prior on the per-band leaf count. The cap is
        # judged on the cold chain (``_update_band_leaf_caps``) but enforced
        # at EVERY temperature so tempering swaps stay within a common prior
        # support. Setting -inf here routes the birth through the existing
        # ``keep`` machinery: it never reaches the likelihood kernel and the
        # bad-accept guard force-rejects it at beta > 0.
        if self._band_leaf_cap is not None:
            num_bands = self.num_bands
            cap_xp = xp.asarray(self._band_leaf_cap)
            flat_all = (
                (band_sorter.temp_inds.astype(xp.int64) * self.nwalkers
                 + band_sorter.walker_inds) * num_bands
                + band_sorter.band_inds
            )
            # Guard empty input: numpy.bincount([]) returns zeros, but
            # CuPy's bincount computes max(x) first and raises on a
            # zero-size array (the zero-leaf search start hits this on GPU).
            _alive_cells = flat_all[band_sorter.inds]
            _nbins = self.ntemps * self.nwalkers * num_bands
            if _alive_cells.shape[0] == 0:
                cell_counts = xp.zeros(_nbins, dtype=xp.int64)
            else:
                cell_counts = xp.bincount(_alive_cells, minlength=_nbins)
            cell_flat = (
                (picked["temp_inds"].astype(xp.int64) * self.nwalkers
                 + picked["walker_inds"]) * num_bands
                + picked["band_inds"]
            )
            over_cap = (
                cell_counts[cell_flat] >= cap_xp[picked["band_inds"]]
            )
            curr_logp[(~alive) & over_cap] = -np.inf

        delta_ll = cp.full_like(logp, -1e300)
        d_h = cp.zeros_like(logp)
        h_h = cp.zeros_like(logp)
        keep = ~cp.isinf(curr_logp)

        # Per-row RJ-factor correction from the F-stat distance-birth proposal
        # (replaces the container's uniform slot-0 term with the lognormal
        # proposal density). Stays None when that path is off.
        _fstat_factor_corr = None

        # One-shot birth-funnel diagnostic (GB_RJ_BIRTH_DEBUG=1): report where
        # births die BEFORE the likelihood eval. A birth reaches scoring only
        # if the GLOBAL prior (self.gpu_priors) accepts its drawn coordinate
        # AND it is in-band AND under the per-band leaf cap. If the F-stat
        # birth container draws coordinates the global prior forbids (range
        # mismatch in dist / Mc / fdot_astro_ratio), every birth is -inf here
        # and the amplitude/phase maximisation below never runs.
        if os.environ.get("GB_RJ_BIRTH_DEBUG"):
            births = ~alive
            nb = int(births.sum())
            prior_inf = int((births & cp.isinf(logp)).sum())
            oob_b = int((births & out_of_band).sum())
            kept_b = int((births & keep).sum())
            fac = band_sorter.factors[ids]
            fb = fac[births]
            logger.info(
                "%s [birth-funnel] births=%d killed{global_prior_inf=%d "
                "out_of_band=%d} kept=%d | factors[births] min=%.3g max=%.3g "
                "nonfinite=%d | logp[births] min=%.3g max=%.3g",
                self.name, nb, prior_inf, oob_b, kept_b,
                float(fb.min()) if nb else 0.0,
                float(fb.max()) if nb else 0.0,
                int((~cp.isfinite(fb)).sum()) if nb else 0,
                float(logp[births].min()) if nb else 0.0,
                float(logp[births].max()) if nb else 0.0,
            )
            # One-time per-sub-prior breakdown: which column of the GLOBAL
            # prior forbids the births? Iterate the ProbDistContainer's
            # (inds, dist) pairs and evaluate each on the birth coordinates.
            if nb and prior_inf and not getattr(self, "_birth_prior_broke_down", False):
                self._birth_prior_broke_down = True
                bparams = params[births]
                pc = self.gpu_priors[self.branch_name]
                for inds, prior_i in pc.priors:
                    try:
                        sub = prior_i.logpdf(bparams[:, list(inds)])
                        sub = cp.asarray(sub)
                        ninf = int((~cp.isfinite(sub)).sum())
                        col_lo = [float(bparams[:, j].min()) for j in inds]
                        col_hi = [float(bparams[:, j].max()) for j in inds]
                        logger.info(
                            "%s [birth-prior] cols=%s -inf=%d/%d | drawn range "
                            "min=%s max=%s | %s",
                            self.name, list(inds), ninf, len(sub),
                            col_lo, col_hi, type(prior_i).__name__,
                        )
                    except Exception as e:  # pragma: no cover - diagnostic only
                        logger.info("%s [birth-prior] cols=%s eval FAILED: %r",
                                    self.name, list(inds), e)

        if bool(keep.any()):
            k_ids = xp.arange(len(ids))[keep]
            birth_k = k_ids[~alive[keep]]
            death_k = k_ids[alive[keep]]

            def _eval(rows, phase_maximize):
                buffer_obj.get_ll(
                    params[rows], slots[rows], slots[rows], N_vals[rows],
                    phase_maximize=phase_maximize,
                )
                # use d_h and h_h to determine birth and death logls by trick with 
                # phase -> -phase
                d_h[rows] = buffer_obj.d_h_out.real
                h_h[rows] = buffer_obj.h_h_out.real
                bad_rows = rows[~buffer_obj.kept_out]
                return bad_rows

            oob_rows = xp.zeros(0, dtype=int)
            if self.rj_fstat_dist_birth and (len(birth_k) or len(death_k)):
                # ---- F-stat distance-birth proposal (step 2) ----
                # Center each birth on the F-stat 4-parameter maximum: draw the
                # distance from a lognormal about ``dist* = amp_from_dist(f0,Mc,
                # 1)/A_max`` (A_max = gbgpu Jaranowski-Krol inversion), and set
                # iota/psi to their F-stat maxima; phi0 is refined against the
                # cell residual by the phase-max in ``_eval``. The proposal
                # density enters the RJ factor (below), so unlike the step-1
                # pin this is detailed-balance-valid. Deaths evaluate the
                # reverse-proposal density at the removed source's own center.
                walker_ref = getattr(self, "_fstat_walker_ref", 0)
                _fstat_factor_corr = cp.zeros(len(ids))
                _log_range = self._log_dist_range(band_sorter)
                if len(birth_k):
                    A_max, phi0_max, iota_max, psi_max, F = self._fstat_dist_centers(
                        model, params[birth_k], walker_ref)
                    ln_center, sigma = self._dist_center_and_width(
                        params[birth_k], A_max, F)
                    z = xp.asarray(cp.random.randn(len(birth_k)))
                    ln_draw = ln_center + sigma * z
                    if _gb_use_distance(self):
                        params[birth_k, 0] = xp.exp(ln_draw)
                    else:
                        params[birth_k, 0] = ln_draw  # slot 0 is lnA already
                    params[birth_k, 4] = xp.cos(iota_max % np.pi)
                    params[birth_k, 5] = psi_max % np.pi
                    params[birth_k, 3] = phi0_max % (2 * np.pi)
                    _bl = self._slot0_log_proposal(params[birth_k, 0], ln_center, sigma)
                    _fstat_factor_corr[birth_k] = -_bl - _log_range
                    # Re-evaluate the global prior at the drawn distance/angles
                    # (the earlier curr_logp used the placeholder draw); f0,
                    # band and leaf-cap gating are unchanged by this overwrite.
                    curr_logp[birth_k] = cp.asarray(
                        self.gpu_priors[self.branch_name].logpdf(params[birth_k]))
                    oob_rows = _eval(birth_k, True)
                    if buffer_obj.phase_angle is not None:
                        params[birth_k, 3] = params[birth_k, 3] - buffer_obj.phase_angle
                if len(death_k):
                    oob_rows = xp.concatenate([oob_rows, _eval(death_k, False)])
                    Ad, _pd, _id, _psd, Fd = self._fstat_dist_centers(
                        model, params[death_k], walker_ref)
                    ln_center_d, sigma_d = self._dist_center_and_width(
                        params[death_k], Ad, Fd)
                    _dl = self._slot0_log_proposal(
                        params[death_k, 0], ln_center_d, sigma_d)
                    _fstat_factor_corr[death_k] = _dl + _log_range
            elif self.phase_maximize and len(birth_k):
                # Maximise the birth phase; deaths keep the true phase.
                oob_rows = _eval(birth_k, True)
                if buffer_obj.phase_angle is not None:
                    params[birth_k, 3] = params[birth_k, 3] - buffer_obj.phase_angle
                if len(death_k):
                    oob_rows = xp.concatenate([oob_rows, _eval(death_k, False)])
            else:
                oob_rows = _eval(k_ids, False)

            # Legacy step-1 amplitude pin: scale the drawn amplitude by the
            # empirical residual ratio ``s = d_h/h_h`` (a 1-parameter fit at the
            # drawn iota/psi). Superseded by the F-stat distance proposal above
            # and only runs when that path is OFF (GB_RJ_FSTAT_DIST_BIRTH=0).
            if (not self.rj_fstat_dist_birth) and self.rj_amp_maximize and len(birth_k):
                hh_b = h_h[birth_k]
                good = hh_b > 0.0
                hh_safe = xp.where(good, hh_b, 1.0)
                s = xp.where(good, d_h[birth_k] / hh_safe, 1.0)
                if _gb_use_distance(self):
                    # A propto 1/dist  ->  dist_new = dist / s
                    params[birth_k, 0] = xp.where(
                        good, params[birth_k, 0] / s, params[birth_k, 0]
                    )
                else:
                    params[birth_k, 0] = xp.where(
                        good, params[birth_k, 0] + xp.log(s), params[birth_k, 0]
                    )
                snr2 = xp.where(good, d_h[birth_k] ** 2 / hh_safe, d_h[birth_k])
                d_h[birth_k] = snr2
                h_h[birth_k] = snr2

            delta_all = xp.where(alive, -d_h - 0.5 * h_h, d_h - 0.5 * h_h)
            delta_ll[keep] = delta_all[keep]
            delta_ll[oob_rows] = -1e300

            # SNR rejection-sampling clamp on births.
            opt_snr = xp.sqrt(xp.maximum(h_h, 0.0))
            reject = (~alive) & keep & (opt_snr < buffer_obj.opt_snr_rej_samp_limit)
            delta_ll[reject] = -1e300

            if os.environ.get("GB_RJ_BIRTH_DEBUG"):
                kb = (~alive) & keep
                nkb = int(kb.sum())
                if nkb:
                    logger.info(
                        "%s [birth-score] kept_births=%d snr_clamped=%d | "
                        "delta_ll[kept] max=%.3g median=%.3g | opt_snr[kept] "
                        "max=%.3g | snr_rej_limit=%.3g",
                        self.name, nkb,
                        int((reject & kb).sum()),
                        float(delta_ll[kb].max()),
                        float(xp.median(delta_ll[kb])),
                        float(opt_snr[kb].max()),
                        float(buffer_obj.opt_snr_rej_samp_limit),
                    )

            self._debug_verify_rj_step(
                buffer_obj, params, alive, slots, N_vals, delta_ll, keep,
                picked, round_i, scheduler,
            )

        beta = band_temps[picked["band_inds"], picked["temp_inds"]]
        factors = band_sorter.factors[ids]
        if _fstat_factor_corr is not None:
            # Swap the container's uniform slot-0 proposal term for the F-stat
            # lognormal distance proposal density (births: -log g; deaths:
            # +log g; the +/-log_range neutralizes the uniform's constant).
            factors = factors + _fstat_factor_corr
        lnpdiff = beta * delta_ll + (curr_logp - prev_logp) + factors
        accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))

        # Coordinates outside the prior can only be accepted at beta == 0;
        # everything else is a bug -> warn and reject.
        bad_mask = (delta_ll <= -1e299) | (curr_logp <= -1e229)
        bad_accepts = accept & bad_mask
        if bool(xp.any(bad_accepts)):
            if bool(xp.any(beta[bad_accepts] != 0.0)) and not (
                "fstat" in self.name or "refit" in self.name
            ):
                logger.warning(
                    f"{self.name}: accepted an out-of-prior RJ coordinate at beta > 0."
                )
            accept[bad_accepts] = False

        rj_seq = getattr(self, "_dbg_rj_seq", None)
        if rj_seq is not None:
            rj_seq["accepted"] = bool(accept[rj_seq["idx"]])

        t_i, w_i, b_i = picked["temp_inds"], picked["walker_inds"], picked["band_inds"]
        prop_counts[0][t_i, w_i, b_i] += 1

        if os.environ.get("GB_RJ_TRACE"):
            # Trace every cold-chain DEATH proposal (accepted or not) plus
            # every accepted cold-chain move. The death delta is
            # -<r|h> - 0.5<h|h>: for a well-fit bright source it must sit
            # near -0.5*SNR^2 and essentially never be accepted at beta=1,
            # so a cold-chain leaf loss means either h_h came back ~0 from
            # the kernel (template dropped: band-edge layer gating /
            # sub-band slab window / stale device index arrays) or the
            # accept bookkeeping raced the residual update. d_h/h_h at
            # proposal time distinguish the two -- compare GPU vs CPU runs
            # of the same seed/config.
            for _k in range(len(ids)):
                if t_i[_k] != 0:
                    continue
                _acc = bool(accept[_k])
                if not (_acc or bool(alive[_k])):
                    continue
                logger.warning(
                    "RJTRACE %s t=%d w=%d b=%d slot=%d f0=%.9e mHz N=%d "
                    "d_h=%.6e h_h=%.6e delta=%.6e beta=%.3e lnp=%.6e "
                    "factors=%.4e curr_lp=%.4e prev_lp=%.4e accept=%d",
                    "DEATH" if alive[_k] else "BIRTH",
                    int(t_i[_k]), int(w_i[_k]), int(b_i[_k]), int(slots[_k]),
                    float(params[_k, 1]), int(N_vals[_k]),
                    float(d_h[_k]), float(h_h[_k]), float(delta_ll[_k]),
                    float(beta[_k]), float(lnpdiff[_k]), float(factors[_k]),
                    float(curr_logp[_k]), float(prev_logp[_k]), int(_acc),
                )

        if bool(accept.any()):
            acc_ids = ids[accept]
            band_sorter.inds[acc_ids] = ~band_sorter.inds[acc_ids]
            # Phase-maximised births carry the rotated phi0 forward.
            band_sorter.coords[acc_ids] = self.periodic.wrap(
                {self.branch_name: params[accept][:, None, :]}, xp=xp
            )[self.branch_name][:, 0]

            ll_change_log[t_i[accept], w_i[accept], b_i[accept]] += delta_ll[accept]
            acc_counts[0][t_i[accept], w_i[accept], b_i[accept]] += 1

            birth_acc = accept & (~alive)
            death_acc = accept & alive
            if bool(birth_acc.any()):
                buffer_obj.add_sources_to_band_buffer(
                    band_sorter.coords[ids[birth_acc]],
                    slots[birth_acc], N_vals[birth_acc],
                    leaf_inds=band_sorter.leaf_inds[ids[birth_acc]],
                )
            if bool(death_acc.any()):
                buffer_obj.remove_sources_from_band_buffer(
                    band_sorter.coords[ids[death_acc]],
                    slots[death_acc], N_vals[death_acc],
                    leaf_inds=band_sorter.leaf_inds[ids[death_acc]],
                )

    def _compute_proposal_cholesky(self, model, band_sorter, ids):
        """Batched Cholesky of the inverse information matrix for ``ids``.

        Domain-symmetric through the fast computation objects:
        FD -> :meth:`GBFDComputations.information_matrix`,
        WDM -> :meth:`GBWDMComputations.information_matrix` (both against the
        parent inverse-covariance rows keyed by walker; the legacy
        SharedMemory ``gb.information_matrix`` path is retired).

        The information matrix comes back in PHYSICAL parameter space; it is mapped to
        the sampling basis with the (numerical, per-source diagonal)
        Jacobian of the transform container, conditioned by the fdot
        rescale, inverted, and factorized.

        Not used by pure-stretch moves (``use_info_mat_proposal=False``,
        e.g. the VGB move); the fdot conditioning column is resolved from
        the transform container's input basis (``self._fdot_col``).
        """
        xp = self.xp
        coords = band_sorter.coords[ids]
        n_src, ndim = coords.shape
        params_phys = self.transform_fn.both_transforms(coords, xp=cp)
        _test_inds = np.asarray(self.parameter_transforms.fill_dict["test_inds"])
        walker_inds = band_sorter.walker_inds[ids].astype(xp.int32)

        # Route the Fisher matrix per shard: the comp is single-shard by
        # contract, but ``model.analysis_container_arr`` may span GPUs (the
        # noise_index rows can live on different devices). Single-shard /
        # CPU holders pass straight through.
        _info_comp = (
            self.gb_fd_comp
            if isinstance(self._basis_settings, FDSettings)
            else self.gb_wdm_comp
        )
        info_phys = _RoutedBandEngine.route_information_matrix(
            _info_comp, model.analysis_container_arr, params_phys,
            inds=_test_inds, noise_index=walker_inds,
        )

        # Conditioning scales for the sampling basis (fdot spans ~1e-13 in
        # sampled units; without the rescale the information matrix inversion is
        # ill-conditioned). The proposal draws in the rescaled coordinates
        # y = x / s and maps back with * s (see in_model_proposal).
        s = xp.ones(ndim)
        if self._fdot_col is not None:
            s[self._fdot_col] = self._fdot_scale
        self._proposal_param_scales = s

        # Numerical diagonal Jacobian d(phys[test_inds[i]]) / d(y_i) through
        # the transform container -- generic in the container's transforms.
        J = xp.zeros((n_src, ndim))
        for i in range(ndim):
            h = 1e-6 * xp.maximum(xp.abs(coords[:, i]), 1e-3)
            up = coords.copy()
            dn = coords.copy()
            up[:, i] += h
            dn[:, i] -= h
            dphys = (
                self.transform_fn.both_transforms(up, xp=cp)[:, _test_inds[i]]
                - self.transform_fn.both_transforms(dn, xp=cp)[:, _test_inds[i]]
            )
            J[:, i] = dphys / (2.0 * h) * s[i]

        info_y = info_phys * J[:, :, None] * J[:, None, :]

        self.mempool.free_all_blocks()
        # Robust inverse-information-matrix factor: near-zero-SNR (prior-drawn) sources
        # give (numerically) singular information matrices. Eigendecompose and clamp the
        # spectrum to a relative floor; B = V diag(lambda^-1/2) satisfies
        # B B^T = inv(info) and is all the Gaussian proposal needs (the
        # proposal shape only -- M-H corrects).
        evals, evecs = xp.linalg.eigh(info_y)
        floor = 1e-10 * xp.maximum(
            xp.abs(evals).max(axis=-1, keepdims=True), 1e-300
        )
        evals = xp.maximum(xp.abs(evals), floor)
        chol = evecs / xp.sqrt(evals)[:, None, :]
        if self._fdot_astro_col is not None:
            # fdot_astro_ratio is likelihood-degenerate with Mc (both enter
            # only through the product fdot_gr(Mc)*(1+r)) and its test_inds
            # target is the dead fddot slot, so the diagonal-Jacobian
            # information matrix carries NO real curvature for it -- the
            # eigen-floor would otherwise hand it an arbitrary huge jump.
            # Zero its proposal row so the info-matrix Gaussian leaves it
            # fixed; the in-model group-stretch component (symmetric, all
            # ndim columns) explores the (Mc, r) ridge. A tailored on-ridge
            # proposal is a documented follow-up.
            chol[:, self._fdot_astro_col, :] = 0.0
        return chol

    def in_model_proposal(self, coords, chol, band_sorter, source_ids, model):
        """Default in-model proposal: group-stretch / info-matrix mix.

        Overridable hook: subclasses provide other proposal components with
        the same ``(new_coords, factors)`` contract (``factors`` are the
        detailed-balance log-factors, zero for symmetric proposals).

        Group stretch is only allowed once the move has completed at least
        one full pass (``self.time >= 1``) and the cold-chain friend table
        exists; it is then drawn with probability ``stretch_probability``
        per repeat round (the info-matrix Cholesky jump otherwise).
        """
        xp = self.xp
        use_stretch = (
            self.stretch_probability > 0.0
            and self.time >= 1
            and getattr(band_sorter, "friend_start_inds", None) is not None
            and bool(np.random.rand() < self.stretch_probability)
        )

        if use_stretch:
            # Friends drawn per source; Eryn's GroupStretchMove supplies the
            # stretch math + (ndim-1)*log(zz) factors through find_friends.
            self._friends_for_stretch = band_sorter.draw_friends(source_ids)
            q, factors = self.get_proposal(
                {self.branch_name: coords[None, :, None, :]},
                model.random,
                s_inds_all={self.branch_name: xp.ones((1, coords.shape[0], 1), dtype=bool)},
            )
            new_coords = q[self.branch_name][0, :, 0, :]
            factors = factors.reshape(-1)
        else:
            # Gaussian jump through the information matrix Cholesky (drawn in the
            # conditioned coordinates y = x / s; mapped back with * s).
            assert chol is not None, (
                "info-matrix branch requires use_info_mat_proposal=True"
            )
            _rand = xp.random.randn(*coords.shape)
            dy = xp.einsum("...ij,...j->...i", chol, _rand)
            new_coords = coords + self.jump_factor * dy * self._proposal_param_scales[None, :]
            factors = xp.zeros(coords.shape[0])   # symmetric draw

        return new_coords, factors

    def _sighet_drift_metrics(self, curr, ref_track, leaf_inds):
        """Per-source drift vs the heterodyne expansion point, PHYSICAL basis.

        Returns ``(drift, damp)``: accumulated carrier-phase drift
        ``2*pi*|df0|*Tobs + pi*|dfdot|*Tobs**2`` (rad) and the amplitude
        ratio ``|ln(A/A_ref)|``. Computed by transforming BOTH coordinate
        sets to physical parameters, so it is branch-agnostic: reduced
        sampling bases (e.g. VGB's 5-col, where f0/sky are per-leaf fills)
        get exactly-zero contributions from the fixed parameters. Reading
        sampling columns directly (the old form) silently misread reduced
        bases -- VGB's column 1 is not f0, which made the drift audit
        report ~1e13 rad of impossible "drift"."""
        li = leaf_inds if self._per_leaf_fill else None
        pc = self.transform_fn.both_transforms(curr, xp=cp, leaf_inds=li)
        pr = self.transform_fn.both_transforms(ref_track, xp=cp, leaf_inds=li)
        Tobs = float(self._basis_settings.Tobs)
        drift = (2.0 * np.pi * cp.abs(pc[:, 1] - pr[:, 1]) * Tobs
                 + np.pi * cp.abs(pc[:, 2] - pr[:, 2]) * Tobs**2)
        damp = cp.abs(cp.log(cp.abs(pc[:, 0]) / cp.abs(pr[:, 0])))
        return drift, damp

    def _sighet_trust_dlna_vec(self, buffer_obj, n):
        """Per-source amplitude gate ``clip(C/snr_ref, dlna_min, dlna_cap)``.

        Reads the reference template power ``h_h_out`` stashed on the
        buffer by the most recent ``get_ll``/``get_add_ll`` call (i.e. the
        block's ``ll_ref`` evaluation, or a refresh's re-basing call for
        the refreshed subset). ``C = 0`` returns the uniform cap."""
        if self.sighet_trust_snr_c <= 0.0:
            return cp.full(n, self.sighet_trust_dlna)
        hh = cp.asarray(buffer_obj.h_h_out).real
        snr_ref = cp.sqrt(cp.clip(hh, 0.0, None))
        return cp.clip(
            self.sighet_trust_snr_c / cp.maximum(snr_ref, 1e-30),
            self.sighet_trust_dlna_min, self.sighet_trust_dlna,
        )

    def _sighet_anchor_phys(self, ref_track, leaf_inds):
        """Anchor-side physical quantities for the trust-region gate.

        Returns ``(|A|, f0, fdot)`` of the heterodyne expansion points so
        the per-repeat gate only transforms the CANDIDATES (the anchor side
        is fixed for the block, modulo mid-block refresh)."""
        li = leaf_inds if self._per_leaf_fill else None
        pr = self.transform_fn.both_transforms(ref_track, xp=cp, leaf_inds=li)
        return cp.abs(pr[:, 0]), pr[:, 1].copy(), pr[:, 2].copy()

    def _run_in_model_repeats(self, model, band_sorter, buffer_obj, band_temps,
                              picked, ll_change_log, prop_counts, acc_counts):
        """``num_repeat_proposals`` in-model rounds on the picked live sources.

        The picked source is first taken OUT of its cell residual, so every
        repeat scores through a plain ``get_add_ll`` against the
        source-free residual (the buffer is not touched between repeats --
        only the tracked coordinates and counters move). After the repeats,
        the final coordinates are written back into the residual and into
        the BandSorter.
        """
        xp = self.xp
        # Read ``inds`` from the MAIN sorter AFTER the RJ step: _run_rj_step
        # flips band_sorter.inds on accepted births/deaths, so this mask
        # includes freshly-born sources (they get the repeat block) and drops
        # freshly-killed ones (their template is already out of the residual).
        # Nested spans under the top-level ``inmodel_repeats`` span (see
        # _ProposeTimer._TOP -- ``inmodel_*`` are reported but excluded from
        # the tracked/untracked accounting, so they don't double-count).
        # This is the breakdown the sig-het GPU work needs: per-block setup
        # (cholesky / sighet_setup) vs the per-repeat scoring kernel
        # (inmodel_get_add_ll) vs host-side MH overhead (inmodel_accept).
        tm = getattr(self, "_prop_timer", None)
        alive = band_sorter.inds[picked["ids"]]
        if not bool(alive.any()):
            return

        ids = picked["ids"][alive]
        slots = picked["slot_index"][alive]
        N_vals = picked["N_vals"][alive]
        t_i = picked["temp_inds"][alive]
        w_i = picked["walker_inds"][alive]
        b_i = picked["band_inds"][alive]
        # Original eryn leaf index of each picked source: threads per-leaf
        # transform fills (Eryn per-leaf fill_dict) through every buffer
        # likelihood/fill call below. Scalar-fill containers ignore it.
        l_i = band_sorter.leaf_inds[ids]
        beta = band_temps[b_i, t_i]

        curr = band_sorter.coords[ids].copy()
        curr[:] = self.periodic.wrap({self.branch_name: curr[:, None, :]}, xp=xp)[self.branch_name][:, 0]

        # Debug 3x3 sequence figures (channels x template/data/residual) at
        # the four buffer moments of this repeat block, for the chosen
        # (walker, band) cell only, once per sampler step.
        seq = self._debug_seq_select(
            buffer_obj, band_sorter, ids, t_i, w_i, b_i, slots, curr)
        if seq is not None:
            seq["snaps"]["before_removal"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])
            # Cell TOTAL DATA (constant across the block): residual +
            # sum of ALL modeled templates of the cell. The figures show
            # total template = data_const - residual, so removing the one
            # picked source appears as +1 signal in the residual and a
            # small dent in the total template.
            _t_tot = self._debug_cell_total_template(
                buffer_obj, band_sorter, seq)
            seq["t_tot"] = _t_tot
            # Combine only when the total template matches the (per-band)
            # residual slab shape; a geometry mismatch degrades to no
            # reconstructed data_const rather than crashing the sampler
            # (debug-only invariant, and _true below often supersedes it).
            _br = seq["snaps"]["before_removal"]
            seq["data_const"] = (
                _br + _t_tot
                if (
                    _t_tot is not None
                    and _br is not None
                    and getattr(_t_tot, "shape", None) == getattr(_br, "shape", None)
                )
                else None
            )
            # TRUE data guard: prefer the injection-data slab (minus
            # non-GB models) over the residual+templates reconstruction --
            # the two coincide unless GB sources are modeled OUTSIDE the
            # traced band (the reconstruction cannot undo those
            # subtractions; the snapshot slice can).
            _true = self._debug_walker_true_data(
                model.analysis_container_arr, seq["walker"])
            if _true is not None:
                seq["data_const"] = _true

        # Take the source out of the cell residual for the whole repeat block.
        with _tspan(tm, "inmodel_removal"):
            buffer_obj.remove_sources_from_band_buffer(curr, slots, N_vals, leaf_inds=l_i)

        if seq is not None:
            seq["snaps"]["after_removal"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])

        # Pure-stretch moves (use_info_mat_proposal=False, e.g. the fixed-
        # dimensional VGB move) never build the info-matrix Cholesky.
        with _tspan(tm, "inmodel_cholesky"):
            chol = (
                self._compute_proposal_cholesky(model, band_sorter, ids)
                if self.use_info_mat_proposal
                else None
            )
        # Per-source likelihood setup for the repeat block (same stage as
        # the proposal cholesky / friend table). Chunked-het / FD engines
        # no-op; a sig-het computation builds its heterodyne reference
        # against the source-free residual HERE and holds it CONSTANT for
        # the whole repeat block, so ll_ref below and every repeat's
        # get_add_ll score through the same likelihood.
        with _tspan(tm, "inmodel_sighet_setup"):
            sighet_active = bool(
                buffer_obj.setup_in_model_likelihood(curr, slots, N_vals, leaf_inds=l_i)
            )
        # Drift-refresh anchor: the sampling-basis coords each source's
        # sig-het reference was built at (see the refresh block below).
        ref_track = curr.copy() if sighet_active else None
        # Trust-region gate cache: anchor-side (|A|, f0, fdot) in the
        # physical basis, so each repeat only transforms the candidates.
        anchor_phys = (
            self._sighet_anchor_phys(ref_track, l_i)
            if sighet_active and self.sighet_trust_dlna > 0.0
            else None
        )
        trust_Tobs = float(self._basis_settings.Tobs)
        with _tspan(tm, "inmodel_ll_ref"):
            ll_ref = buffer_obj.get_add_ll(curr, slots, slots, N_vals, leaf_inds=l_i)
            curr_prior = cp.asarray(self.gpu_priors[self.branch_name].logpdf(curr))

        # Cold-chain <d|h>, <h|h> capture on the sorter's flat source storage
        # (leaf labels are only final after the repack in _write_back_state):
        # seeded from the block-reference get_add_ll (which just filled
        # buffer_obj.d_h_out/h_h_out for ``curr``), updated at each accepted
        # move below. Zero extra likelihood evaluations. MUST run before the
        # sig-het anchor check below — its exact-path re-evaluation
        # overwrites d_h_out/h_h_out.
        if getattr(buffer_obj, "d_h_out", None) is not None:
            if getattr(self, "_sorter_dh", None) is None:
                n_src = band_sorter.inds.shape[0]
                self._sorter_dh = cp.full(n_src, np.nan)
                self._sorter_hh = cp.full(n_src, np.nan)
            self._sorter_dh[ids] = cp.asarray(buffer_obj.d_h_out).real
            self._sorter_hh[ids] = cp.asarray(buffer_obj.h_h_out).real

        # Per-source SNR-scaled amplitude gate (see the ctor comment):
        # snr_ref = sqrt(h_h) at the anchor, stashed by the ll_ref
        # evaluation just above. Vectorized once per block; the repeat
        # loop compares candidates against ``trust_dlna[sl]``.
        trust_dlna = None
        if anchor_phys is not None:
            trust_dlna = self._sighet_trust_dlna_vec(buffer_obj, len(ids))
        # Anchor check (debug knob; see ctor comment): sig-het vs exact at
        # the block anchor itself, where the ratio is exactly 1. Rebuilds
        # the reference afterwards (fresh == patched is bit-exact).
        if sighet_active and self.sighet_anchor_check:
            with _tspan(tm, "inmodel_anchor_check"):
                buffer_obj.clear_in_model_likelihood()
                _ll_ex0 = buffer_obj.get_add_ll(
                    curr, slots, slots, N_vals, leaf_inds=l_i)
                buffer_obj.setup_in_model_likelihood(
                    curr, slots, N_vals, leaf_inds=l_i)
            _e0 = cp.abs(ll_ref - _ll_ex0)
            _i0 = int(cp.argmax(_e0))
            _f0_0 = float(_to_numpy(self.transform_fn.both_transforms(
                curr[_i0:_i0 + 1], xp=cp,
                leaf_inds=(l_i[_i0:_i0 + 1]
                           if self._per_leaf_fill else None),
            )[0, 1]))
            logger.info(
                f"{self.name}: sig-het ANCHOR check ({len(ids)} sources): "
                f"|dll@anchor| max={float(_e0.max()):.3e} "
                f"median={float(cp.median(_e0)):.3e}; worst: "
                f"temp={int(t_i[_i0])} walker={int(w_i[_i0])} "
                f"band={int(b_i[_i0])} f0={_f0_0:.6e} Hz "
                f"ll_het={float(ll_ref[_i0]):.3e} "
                f"ll_exact={float(_ll_ex0[_i0]):.3e}"
            )
        if tm is not None:
            # Per-block scale, so a wall time can be read as a per-source /
            # per-repeat cost without cross-referencing the run config.
            tm.count("inmodel_sources", len(ids))
            tm.count("inmodel_blocks")

        n4 = (N_vals / 4).astype(int)
        lo_bin = (buffer_obj.frequency_lims[0][slots] / self.df).astype(int)
        hi_bin = (buffer_obj.frequency_lims[1][slots] / self.df).astype(int)

        # Sequential red-blue repeats (VGB): each repeat runs as TWO
        # half-sweeps split by walker parity, with the sorter coords synced
        # to ``curr`` before each half so the stretch complement (read from
        # ``band_sorter.coords`` inside ``in_model_proposal``) is always the
        # CURRENT opposite half -- blue updates red, then blue moves against
        # the UPDATED red, exactly eryn's ``RedBlueMove`` split structure.
        # Each half-sweep is then an invariant kernel and the repeat count
        # is a cost knob, not a bias knob. The default single full-batch
        # sweep stays for the GB moves, whose group-stretch friend table /
        # info-matrix Cholesky complements are block-start structures by
        # design (repeats DO matter there).
        if self.sequential_parity_repeats:
            halves = [xp.where(w_i % 2 == p)[0] for p in (0, 1)]
            halves = [h for h in halves if h.size > 0]
        else:
            halves = [None]

        for move_i in range(self.num_repeat_proposals):
          for sub in halves:
            if sub is None:
                sl = slice(None)          # full batch (original behavior)
                n_sub = len(ids)
            else:
                sl = sub
                n_sub = int(sub.size)
                # Complement <- current state (including the other half's
                # accepted moves) for this half's proposal.
                band_sorter.coords[ids] = curr

            with _tspan(tm, "inmodel_proposal"):
                new, factors = self.in_model_proposal(
                    curr[sl], None if chol is None else chol[sl],
                    band_sorter, ids[sl], model)
                new[:] = self.periodic.wrap({self.branch_name: new[:, None, :]}, xp=xp)[self.branch_name][:, 0]

            with _tspan(tm, "inmodel_prior"):
                new_logp = cp.asarray(self.gpu_priors[self.branch_name].logpdf(new))
                # In-model steps stay within +- N/4 bins of the current source
                # and inside the band window (widened by N/4). Skipped when f0 is
                # a per-leaf fill (not sampled): the proposal cannot move it.
                if self._f0_col is not None:
                    _fc = self._f0_col
                    n4_s, lo_s, hi_s = n4[sl], lo_bin[sl], hi_bin[sl]
                    new_bin = cp.abs(new[:, _fc] / 1e3 / self.df).astype(int)
                    new_logp[
                        (cp.abs(new[:, _fc] / 1e3 - curr[sl][:, _fc] / 1e3) / self.df).astype(int) > n4_s
                    ] = -np.inf
                    new_logp[new_bin < lo_s - n4_s] = -np.inf
                    new_logp[new_bin > hi_s + n4_s] = -np.inf

                # Sig-het TRUST REGION: reject candidates outside the
                # expansion's validity region around the block anchor
                # (physical |dlnA| / carrier-phase gates; see the ctor
                # comment for thresholds + MH-validity). Gated rows drop
                # out of ``keep`` below, so they also skip the ll kernel.
                if anchor_phys is not None:
                    _pc = self.transform_fn.both_transforms(
                        new, xp=cp,
                        leaf_inds=l_i[sl] if self._per_leaf_fill else None,
                    )
                    _damp_n = cp.abs(cp.log(
                        cp.abs(_pc[:, 0]) / anchor_phys[0][sl]))
                    _drift_n = (
                        2.0 * np.pi * cp.abs(_pc[:, 1] - anchor_phys[1][sl])
                        * trust_Tobs
                        + np.pi * cp.abs(_pc[:, 2] - anchor_phys[2][sl])
                        * trust_Tobs**2
                    )
                    new_logp[
                        (_damp_n > trust_dlna[sl])
                        | (_drift_n > self.sighet_trust_dphase)
                    ] = -np.inf

                keep = ~cp.isinf(new_logp)
            new_ll = cp.full(n_sub, -1e300)
            slots_s, N_s, l_s = slots[sl], N_vals[sl], l_i[sl]
            # THE per-repeat scoring call: the sig-het fused in-kernel
            # likelihood when a reference is active, the chunked-het/FD
            # engine otherwise. This span is the headline number for the
            # in-model GB/GB speedup work.
            with _tspan(tm, "inmodel_get_add_ll"):
                if bool(keep.any()):
                    new_ll[keep] = buffer_obj.get_add_ll(
                        new[keep], slots_s[keep], slots_s[keep], N_s[keep],
                        phase_maximize=self.phase_maximize,
                        leaf_inds=l_s[keep],
                    )
                    if self.phase_maximize and buffer_obj.phase_angle is not None:
                        new[keep, self._phi0_col] = (
                            new[keep, self._phi0_col] - buffer_obj.phase_angle
                        )
                        new[keep] = self.periodic.wrap(
                            {self.branch_name: new[keep][:, None, :]}, xp=xp
                        )[self.branch_name][:, 0]
            if tm is not None:
                tm.count("inmodel_repeat_calls")

            delta_ll = new_ll - ll_ref[sl]
            # Host-side MH bookkeeping. On CuPy every ``bool(...any())`` here
            # is a device sync, so this span is the launch-overhead signal:
            # if it rivals inmodel_get_add_ll the block is host-bound (too
            # few sources per launch), not kernel-bound.
            with _tspan(tm, "inmodel_accept"):
                lnpdiff = beta[sl] * delta_ll + (new_logp - curr_prior[sl]) + factors
                accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))

                bad_mask = (new_ll <= -1e299) | (new_logp <= -1e229)
                bad_accepts = accept & bad_mask
                if bool(xp.any(bad_accepts)):
                    if bool(xp.any(beta[sl][bad_accepts] != 0.0)):
                        logger.warning(
                            f"{self.name}: accepted an out-of-prior in-model "
                            "coordinate at beta > 0."
                        )
                    accept[bad_accepts] = False

                prop_counts[1][t_i[sl], w_i[sl], b_i[sl]] += 1
                if bool(accept.any()):
                    # Global positions of the accepted movers: a boolean mask
                    # on the full path, the half's index array otherwise.
                    gi = accept if sub is None else sub[accept]
                    curr[gi] = new[accept]
                    ll_ref[gi] = new_ll[accept]
                    curr_prior[gi] = new_logp[accept]
                    ll_change_log[t_i[gi], w_i[gi], b_i[gi]] += delta_ll[accept]
                    acc_counts[1][t_i[gi], w_i[gi], b_i[gi]] += 1
                    if (
                        getattr(self, "_sorter_dh", None) is not None
                        and getattr(buffer_obj, "d_h_out", None) is not None
                    ):
                        # d_h_out/h_h_out hold the per-repeat get_add_ll
                        # outputs for the ``keep`` subset; select the
                        # accepted rows within it
                        _acc_kept = accept[keep]
                        self._sorter_dh[ids[gi]] = cp.asarray(
                            buffer_obj.d_h_out
                        ).real[_acc_kept]
                        self._sorter_hh[ids[gi]] = cp.asarray(
                            buffer_obj.h_h_out
                        ).real[_acc_kept]

            self._debug_verify_in_model(
                buffer_obj, curr[sl], new, slots_s, N_s, delta_ll, keep,
                (asnumpy(t_i[sl]), asnumpy(w_i[sl]), asnumpy(b_i[sl])), move_i,
            )

            # Sig-het drift refresh: every ``sighet_refresh_every``
            # repeats, re-anchor the references of the sources that
            # walked too far from their expansion point. The test is
            # pure parameter arithmetic (no kernel call): accumulated
            # carrier-phase drift 2*pi*|df0|*Tobs + pi*|dfdot|*Tobs^2
            # plus an amplitude-ratio guard. Refreshed sources get their
            # reference PATCHED in place (only those coefficient blocks
            # rebuild) and their ll_ref re-based against the new
            # reference so the MH deltas never mix references.
            if (
                sighet_active
                # once per REPEAT: only after the last parity half-sweep
                # (halves == [None] on the full-batch path, so this is
                # always true there).
                and sub is halves[-1]
                and self.sighet_refresh_every > 0
                and (move_i + 1) % self.sighet_refresh_every == 0
                and move_i + 1 < self.num_repeat_proposals
            ):
                drift, damp = self._sighet_drift_metrics(curr, ref_track, l_i)
                far = (drift > self.sighet_refresh_dphase) | (damp > np.log(2.0))
                # Hot cells keep their stale reference: the ll error is
                # beta-suppressed and each refresh is a full setup.
                far = far & (beta >= self.sighet_refresh_min_beta)
                if bool(far.any()):
                    with _tspan(tm, "inmodel_sighet_refresh"):
                        buffer_obj.setup_in_model_likelihood(
                            curr[far], slots[far], N_vals[far], leaf_inds=l_i[far]
                        )
                        ll_ref[far] = buffer_obj.get_add_ll(
                            curr[far], slots[far], slots[far], N_vals[far],
                            leaf_inds=l_i[far],
                        )
                    ref_track[far] = curr[far]
                    # Re-anchor the trust-region cache with the refreshed
                    # references (refresh is rare; full recompute is cheap).
                    # The re-basing get_add_ll above stashed h_h for the
                    # refreshed subset, so the SNR-scaled gate re-anchors
                    # from the same call.
                    if anchor_phys is not None:
                        anchor_phys = self._sighet_anchor_phys(ref_track, l_i)
                        trust_dlna[far] = self._sighet_trust_dlna_vec(
                            buffer_obj, int(far.sum()))
                    if tm is not None:
                        tm.count("inmodel_refreshed_sources", int(far.sum()))
                    logger.debug(
                        f"{self.name}: sig-het reference refresh for "
                        f"{int(far.sum())}/{len(ids)} sources at repeat "
                        f"{move_i + 1}."
                    )

        # End-of-block drift AUDIT (sighet_drift_check / GB_SIGHET_DRIFT_CHECK):
        # with the fixed-reference policy (sighet_refresh_every=0) this logs
        # how far each source walked from its heterodyne expansion point over
        # the block -- same parameter-space metric the legacy refresh gated
        # on -- WITHOUT changing the sampling. Pure arithmetic, no kernel.
        _audit = sighet_active and self.sighet_drift_check
        if _audit:
            drift, damp = self._sighet_drift_metrics(curr, ref_track, l_i)
            n_over = int((drift > self.sighet_refresh_dphase).sum())
            _gate = (
                f" gate=[{float(trust_dlna.min()):.2f}..{float(trust_dlna.max()):.2f}]"
                if trust_dlna is not None else ""
            )
            logger.info(
                f"{self.name}: sig-het end-of-block drift ({len(ids)} sources, "
                f"{self.num_repeat_proposals} repeats): phase max="
                f"{float(drift.max()):.3e} median={float(cp.median(drift)):.3e} rad, "
                f"{n_over} over dphase={self.sighet_refresh_dphase}; "
                f"|dlnA| max={float(damp.max()):.3e}.{_gate}"
            )
            # The sig-het delta the CHAIN actually used at the final coords
            # (tracked ll_ref), captured before the engine reverts.
            _ll_het_final = ll_ref.copy()

        # Repeat block over: deactivate the per-source likelihood setup so
        # everything outside the block (RJ, removal, fills) scores through
        # the standard engine path again.
        buffer_obj.clear_in_model_likelihood()

        # End-of-block LIKELIHOOD accuracy AUDIT (same knob as the drift
        # audit): re-score the block's FINAL coordinates through the EXACT
        # engine -- after clear_in_model the buffer routes to the chunked
        # delegate natively -- and compare against the sig-het value the MH
        # chain used. This is the accuracy tracker for the fixed-reference
        # policy at the chain's actual operating points: |dll| is directly
        # comparable to the het budget (dlnL ~ SNR^2 * mm), reported for all
        # temps and for the COLD chain separately (hot walkers legitimately
        # roam where the linearization is worst and beta suppresses the
        # error's effect there). Costs ONE exact batched call per block.
        if _audit:
            with _tspan(tm, "inmodel_accuracy_check"):
                _ll_exact = buffer_obj.get_add_ll(
                    curr, slots, slots, N_vals,
                    phase_maximize=self.phase_maximize, leaf_inds=l_i)
            _err = cp.abs(_ll_het_final - _ll_exact)
            _cold = beta > 0.999
            _n_c = int(_cold.sum())
            _cmax = float(_err[_cold].max()) if _n_c else float("nan")
            _cmed = float(cp.median(_err[_cold])) if _n_c else float("nan")
            logger.info(
                f"{self.name}: sig-het end-of-block ll AUDIT vs exact "
                f"({len(ids)} sources): |dll| max={float(_err.max()):.3e} "
                f"median={float(cp.median(_err)):.3e}; COLD ({_n_c}): "
                f"max={_cmax:.3e} median={_cmed:.3e}."
            )
            # Name the worst COLD offender when it exceeds O(1) in lnL:
            # which source/walker still breaks the fixed-reference
            # expansion, and how far it walked from its anchor. ``drift``
            # / ``damp`` are the drift-audit metrics computed above.
            if _n_c and _cmax > 1.0:
                _ci = cp.where(_cold)[0]
                _ic = int(_ci[int(cp.argmax(_err[_cold]))])
                _f0 = float(_to_numpy(self.transform_fn.both_transforms(
                    curr[_ic:_ic + 1], xp=cp,
                    leaf_inds=(l_i[_ic:_ic + 1]
                               if self._per_leaf_fill else None),
                )[0, 1]))
                logger.warning(
                    f"{self.name}: ll AUDIT worst cold offender: "
                    f"temp={int(t_i[_ic])} walker={int(w_i[_ic])} "
                    f"band={int(b_i[_ic])} f0={_f0:.6e} Hz "
                    f"|dll|={float(_err[_ic]):.3e} "
                    f"dlnA={float(damp[_ic]):.3e} "
                    f"dphase={float(drift[_ic]):.3e} rad "
                    f"ll_exact={float(_ll_exact[_ic]):.3e}"
                )

        # Final coordinates back into the residual and the sorter.
        band_sorter.coords[ids] = curr
        if seq is not None:
            seq["snaps"]["before_addback"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])
            # Final get_ll value for the traced source (post-repeats), for
            # the addback delta-ll cross-check in the sequence figures.
            seq["ll_ref_final"] = float(_to_numpy(ll_ref)[seq["idx"]])
        with _tspan(tm, "inmodel_addback"):
            buffer_obj.add_sources_to_band_buffer(curr, slots, N_vals, leaf_inds=l_i)
        if seq is not None:
            seq["snaps"]["after_addback"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])
            _idx_sl = slice(seq["idx"], seq["idx"] + 1)
            seq["f0_new"] = float(_to_numpy(
                self.transform_fn.both_transforms(
                    curr[_idx_sl], xp=cp,
                    leaf_inds=l_i[_idx_sl] if self._per_leaf_fill else None,
                )[0, 1]))
            self._debug_plot_band_sequence(buffer_obj, seq)

    def _permute_walkers_for_swaps(self):
        """One walker permutation for a (band, temp) tempering row.

        Global permutation on a single device; per-device-block permutation
        when the model ACA shards walkers across GPUs (set by
        ``run_tempering``), so a swap pair's parent walkers always share a
        device (parallel-resources plan P1). Row positions in a block only
        ever hold walkers from that block, so every adjacent-temperature
        pair within a row is device-local.
        """
        groups = getattr(self, "_tempering_walker_groups", None)
        if not groups:
            return cp.random.permutation(cp.arange(self.nwalkers))
        out = cp.empty(self.nwalkers, dtype=int)
        for g in groups:
            g_dev = cp.asarray(g)
            out[g_dev] = g_dev[cp.random.permutation(len(g))]
        return out

    def _tempering_swap_grid(self, band_sorter, start):
        """Permuted (band, walker, temp) cell grid for one tempering parity.

        Interior bands only (the edge bands host no swaps), every
        temperature, and an independent random walker permutation per
        (band, temp) -- adjacent temperature columns of a grid row are the
        cells whose templates may exchange. Only the ``start``-parity
        interior bands are kept.

        Returns ``(band_index, temp_index, walkers_permuted, special_index,
        num_bands_unit)``; the first four are shaped
        ``(bands_this_parity, nwalkers, ntemps)``.
        """
        if self.num_bands == 1:
            num_bands_tempered = 1
            band_index_arr = cp.arange(1)
        else:
            num_bands_tempered = self.num_bands - 2
            band_index_arr = cp.arange(1, self.num_bands - 1)

        num_bands_unit = np.arange(num_bands_tempered)[start::2].shape[0]

        walkers_permuted = (
            cp.asarray(
                [
                    self._permute_walkers_for_swaps()
                    for _ in range(self.ntemps * num_bands_tempered)
                ]
            )
            .reshape(num_bands_tempered, self.ntemps, self.nwalkers)
            .transpose(0, 2, 1)[start::2]
        )
        temp_index = (
            cp.repeat(cp.arange(self.ntemps), num_bands_tempered * self.nwalkers)
            .reshape(self.ntemps, num_bands_tempered, self.nwalkers)
            .transpose(1, 2, 0)[start::2]
        )
        band_index = (
            cp.repeat(band_index_arr, self.ntemps * self.nwalkers)
            .reshape(num_bands_tempered, self.ntemps, self.nwalkers)
            .transpose(0, 2, 1)[start::2]
        )
        special_index = band_sorter.get_special_band_index(
            temp_index, walkers_permuted, band_index
        )
        return band_index, temp_index, walkers_permuted, special_index, num_bands_unit

    def _adapt_band_temps(self, band_temps, band_swaps_accepted, band_swaps_proposed):
        """Per-band temperature-ladder adaptation, in place on ``band_temps``.

        Hyperbolic-decay adjustment of the inverse-temperature ladder from
        the just-collected swap acceptance ratios (hottest and coldest
        chains pinned) -- the standard eryn/ptemcee adaptation applied
        band-by-band. No-op on the first proposal (``self.time == 0``).

        TODO: change temperature adaptation.
        """
        if self.time <= 0:
            return
        # Edge bands never receive swap proposals (interior-bands-only grid),
        # so guard the 0/0: an unproposed (band, pair) adapts with ratio 0
        # instead of propagating NaN into the ladder (at ntemps > 2 a NaN
        # here corrupts band_temps for the edge bands' middle temps, which
        # then NaN-poisons every acceptance in those bands).
        _prop_safe = cp.maximum(band_swaps_proposed, 1)
        ratios = (band_swaps_accepted / _prop_safe).T
        betas0 = band_temps.copy().T
        betas1 = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.temperature_control.adaptation_lag / (
            self.time + self.temperature_control.adaptation_lag
        )
        kappa = decay / self.temperature_control.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = cp.diff(1 / betas1[:-1], axis=0)

        deltaTs *= cp.exp(dSs)
        betas1[1:-1] = 1 / (cp.cumsum(deltaTs, axis=0) + 1 / betas1[0])

        dbetas = betas1 - betas0
        band_temps += self.xp.asarray(dbetas.T)

    def run_tempering(self, model, state, band_sorter, band_temps):
        # Per-GPU temperature permutation (parallel-resources plan P1): when
        # the model ACA splits the cold-chain walkers across devices, swap
        # partners are drawn WITHIN each device's walker block so no swap
        # couples residuals on different GPUs. Walkers are exchangeable, so
        # a device-local permutation is still a correct swap kernel; single
        # device -> one global block (behavior unchanged).
        _aca = getattr(model, "analysis_container_arr", None)
        _splits = getattr(_aca, "gpu_splits", None)
        self._tempering_walker_groups = (
            [np.asarray(asnumpy(s), dtype=int) for s in _splits]
            if _splits is not None and len(_splits) > 1
            else None
        )
        ll_change_log_temp = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))

        band_swaps_accepted = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)

        units = 2
        tmp_start = np.random.randint(units)
        for tmp in range(units):
            remainder = (tmp_start + tmp) % units
            start = remainder
            # start == 0 pairs with bool_remainder 1 because tempering
            # begins at band 1 (the interior bands).
            bool_remainder = 1 if start == 0 else 0

            with _tspan(getattr(self, "_prop_timer", None), "temper_open_close"):
                self.remove_cold_chain_sources_from_residual(
                    model,
                    band_sorter,
                    extra_bool=(band_sorter.band_inds % 2 == bool_remainder),
                )

            (band_index, temp_index, walkers_permuted, special_index,
             num_bands_unit) = self._tempering_swap_grid(band_sorter, start)

            # Tempering chunk size as a CELL budget (rows x ntemps), not a
            # row count: the historic hardcoded 200 rows meant 200*ntemps
            # cells, which scaled the buffer (and its host-side staging)
            # linearly with the temperature ladder -- a 24-temp run built
            # 4800-cell chunks and OOM-killed a 64 GB host allocation
            # (2026-07-23). Default 1200 cells == the validated 6-temp size.
            _cell_budget = int(os.environ.get("GB_TEMPER_PRELOAD_CELLS", "1200"))
            num_bands_preload_temp = max(1, _cell_budget // self.ntemps)
            num_bands_run = 0
            while num_bands_run < self.nwalkers * num_bands_unit:
                start_ind = num_bands_run
                end_ind = start_ind + num_bands_preload_temp

                band_inds_now = band_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                walker_inds_now = walkers_permuted.reshape(-1, self.ntemps)[
                    start_ind:end_ind
                ].copy()
                special_inds_now = special_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                special_inds_now_flat = special_inds_now.flatten()

                with _tspan(getattr(self, "_prop_timer", None), "temper_buffer"):
                    buffer_obj = self._cached_get_buffer(
                        band_sorter, model.analysis_container_arr,
                        special_inds_now_flat,
                        use_template_arr=True,
                    )

                with _tspan(getattr(self, "_prop_timer", None), "temper_swap_score"):
                    current_lls = buffer_obj.band_likelihoods(source_only=True).reshape(-1, self.ntemps)
                current_lls_orig = current_lls.copy()
                for t in range(self.ntemps)[1:][::-1]:
                    i1 = t
                    i2 = t - 1

                    # Buffer slots interleave temperatures: column t of a
                    # grid row is slot (row * ntemps + t).
                    buffer_i1 = cp.arange(buffer_obj.num_bands_now)[i1 :: self.ntemps]
                    buffer_i2 = cp.arange(buffer_obj.num_bands_now)[i2 :: self.ntemps]

                    buffer_obj.swap_template_slots(buffer_i1, buffer_i2)

                    # TODO: add indices because not every likelihood is needed
                    # TODO: C-side vectorized temperature-pair swap kernel
                    # (batch the template exchange + per-cell likelihoods).
                    with _tspan(getattr(self, "_prop_timer", None), "temper_swap_score"):
                        new_lls = buffer_obj.band_likelihoods(source_only=True).reshape(-1, self.ntemps)[
                            :, i2 : i1 + 1
                        ]
                    old_lls = current_lls[:, i2 : i1 + 1]

                    beta1 = band_temps[(band_inds_now[:, 0], i1)]
                    beta2 = band_temps[(band_inds_now[:, 0], i2)]

                    paccept = beta1 * (new_lls[:, 1] - old_lls[:, 1]) + beta2 * (
                        new_lls[:, 0] - old_lls[:, 0]
                    ) # ! this is changed because it think this was wrong, below is the previous paccept (comparing with paccept in paper, it should now be good)
                    # paccept = bi * (band_here_i1->swapped_like - band_here_i->current_like) + bi1 * (band_here_i->swapped_like - band_here_i1->current_like);

                    raccept = cp.log(cp.random.uniform(size=paccept.shape))
                    sel = paccept > raccept

                    current_lls[sel, i2 : i1 + 1] = new_lls[sel]

                    # Reverse the swaps that were not accepted.
                    buffer_obj.swap_template_slots(buffer_i1[~sel], buffer_i2[~sel])

                    # bincount accumulation: several grid rows share a band
                    # (one per walker), and fancy-index ``+=`` collapses
                    # duplicate indices (arr[[1,1,1]] += 1 increments ONCE).
                    # The counters must add one per row, not one per band.
                    # NOTE: guard empty inputs — numpy.bincount([]) returns
                    # zeros, but CuPy's bincount computes max(x) first and
                    # raises on a zero-size array (no accepted swaps in a
                    # chunk is the common case).
                    _nb_tot = band_swaps_accepted.shape[0]
                    _acc_bands = band_inds_now[sel, 0]
                    if _acc_bands.size:
                        band_swaps_accepted[:, i2] += cp.bincount(
                            _acc_bands, minlength=_nb_tot
                        ).astype(band_swaps_accepted.dtype)
                    if band_inds_now.size:
                        band_swaps_proposed[:, i2] += cp.bincount(
                            band_inds_now[:, 0], minlength=_nb_tot
                        ).astype(band_swaps_proposed.dtype)

                    # Accepted cells trade their (temp, walker) labels in the
                    # sorter so the sources follow their templates.
                    specials_i1 = band_sorter.get_special_band_index(
                        i1, walker_inds_now[sel, i1], band_inds_now[sel, i1]
                    )
                    specials_i2 = band_sorter.get_special_band_index(
                        i2, walker_inds_now[sel, i2], band_inds_now[sel, i2]
                    )
                    band_sorter.exchange_cell_labels(
                        specials_i1, i1, walker_inds_now[sel, i1],
                        specials_i2, i2, walker_inds_now[sel, i2],
                        bands=band_inds_now[sel, i2],
                    )

                diffs = current_lls - current_lls_orig
                # ``=`` (not ``+=``): each (temp, walker, band) cell is
                # visited exactly once per tempering pass.
                ll_change_log_temp[
                    (
                        buffer_obj.unique_band_combos[:, 0],
                        buffer_obj.unique_band_combos[:, 1],
                        buffer_obj.unique_band_combos[:, 2],
                    )
                ] = diffs.flatten()
                num_bands_run += num_bands_preload_temp

            # ll_before3 = model.analysis_container_arr.likelihood()
            with _tspan(getattr(self, "_prop_timer", None), "temper_open_close"):
                self.add_cold_chain_sources_to_residual(
                    model,
                    band_sorter,
                    extra_bool=(band_sorter.band_inds % 2 == bool_remainder),
                )
            # ll_after3 = model.analysis_container_arr.likelihood()

        self._adapt_band_temps(band_temps, band_swaps_accepted, band_swaps_proposed)

        # TODO Ask michael what this is about print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum_temp = ll_change_log_temp.sum(axis=-1)

        return ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed

    def _write_back_state(self, new_state, band_sorter) -> None:
        """Repack the sorter's live sources into ``new_state.branches['gb']``.

        Leaves are re-indexed densely per (temp, walker) in frequency order:
        live sources are ranked by the composite key
        ``(temp * nwalkers + walker) * 1e6 + f0`` and numbered ``0..n-1``
        within each (temp, walker) block. ``inds`` is rebuilt from scratch
        (all ``False``, then ``True`` at the repacked leaves), so RJ
        births/deaths and tempering walker reassignments all land here.

        TODO: NEED TO PROPERLY MOVE SUPPLEMENTAL INFO BASED ON OLD LEAVES
        (``inds_old`` below is the source-side index for that move).

        With ``preserve_leaf_identity`` (fixed-dimensional branches whose
        leaf i IS a specific physical source, e.g. VGBs), sources go back
        to their ORIGINAL leaf index at their CURRENT (temp, walker) labels
        (tempering only relabels temp/walker; leaves never move), so the
        per-leaf transform fills stay attached to the right source.
        """
        alive = band_sorter.inds
        # the working branch (the sub-state's tempered ensemble; writes land
        # on the sub-state arrays through the shared-memory view)
        work = self._work_branch(new_state)

        if self.preserve_leaf_identity:
            inds_new = (
                _to_numpy(band_sorter.temp_inds[alive]),
                _to_numpy(band_sorter.walker_inds[alive]),
                _to_numpy(band_sorter.leaf_inds[alive]),
            )
            work.coords[inds_new] = _to_numpy(band_sorter.coords[alive])
            work.inds[:] = False
            work.inds[inds_new] = True
            self._scatter_leaf_products(new_state, alive, inds_new)
            self._sync_cold_row(new_state)
            return
        special_indices_finish = (
            band_sorter.temp_inds[alive] * self.nwalkers
            + band_sorter.walker_inds[alive]
        ) * int(1e6) + band_sorter.coords[alive, 1]
        special_inds_temp_walker = (
            band_sorter.temp_inds[alive] * self.nwalkers
            + band_sorter.walker_inds[alive]
        )
        sorted_inds = cp.argsort(special_indices_finish)

        uni, uni_inds, uni_inverse, uni_counts = cp.unique(
            special_inds_temp_walker[sorted_inds],
            return_index=True,
            return_counts=True,
            return_inverse=True,
        )

        leaf_inds_new_tmp = cp.arange(special_indices_finish.shape[0]) - uni_inds[uni_inverse]
        leaf_inds_new = cp.zeros_like(leaf_inds_new_tmp)
        leaf_inds_new[sorted_inds] = leaf_inds_new_tmp

        inds_new = (
            _to_numpy(band_sorter.temp_inds[alive]),
            _to_numpy(band_sorter.walker_inds[alive]),
            _to_numpy(leaf_inds_new),
        )
        inds_old = (
            _to_numpy(band_sorter.orig_temp_inds[alive]),
            _to_numpy(band_sorter.orig_walker_inds[alive]),
            _to_numpy(band_sorter.orig_leaf_inds[alive]),
        )
        work.coords[inds_new] = _to_numpy(band_sorter.coords[alive])
        work.inds[:] = False
        # turn on all the ones that are there
        work.inds[inds_new] = True
        # work.branch_supplemental[inds_new] = state.branches[self.branch_name].branch_supplemental[inds_old]
        self._scatter_leaf_products(new_state, alive, inds_new)
        self._sync_cold_row(new_state)

    def _scatter_leaf_products(self, new_state, alive, inds_new) -> None:
        """Write the captured cold-chain per-leaf ``<d|h>``/``<h|h>`` into the sub-state.

        The capture lives on the sorter's flat source storage
        (``self._sorter_dh``/``_hh``, filled in ``_run_in_model_repeats``);
        here — after the leaf repack — the alive sources' values land at
        their FINAL (walker, leaf) positions, cold chain only.
        """
        sub_states = getattr(new_state, "sub_states", None) or {}
        sub = sub_states.get(self.branch_name)
        if (
            getattr(self, "_sorter_dh", None) is None
            or sub is None
            or getattr(sub, "d_h", None) is None
        ):
            return
        t_new, w_new, leaf_new = inds_new
        cold = t_new == 0
        dh_alive = _to_numpy(self._sorter_dh[alive])
        hh_alive = _to_numpy(self._sorter_hh[alive])
        sub.d_h[:] = np.nan
        sub.h_h[:] = np.nan
        sub.d_h[w_new[cold], leaf_new[cold]] = dh_alive[cold]
        sub.h_h[w_new[cold], leaf_new[cold]] = hh_alive[cold]

    def _band_residual_lls(self, acs):
        """Per-band cold-walker residual ll ``-1/2 <r|r>`` from the parent ACA.

        Returns a host ``(nwalkers, num_bands)`` array. The parent ACA holds
        one AC per COLD-chain walker, so this is exactly the per-band null
        the leaf-cap convergence test needs. Shard-aware: each per-GPU (or
        per-CPU-split) slab is reduced on its owning device via a per-bin ll
        followed by a cumulative-sum band reduction (no per-band kernel
        loop). Also stores ``self._band_dof`` -- the real-dof count per band
        used to scale the convergence tolerance.
        """
        xp = self.xp
        bs = self._basis_settings
        be = _to_numpy(self.band_edges)
        num_bands = self.num_bands
        norm = 4.0 * float(bs.differential_component)
        is_wdm = isinstance(bs, WDMSettings)
        nchannels = int(acs.nchannels)
        # XYZ runs carry the full cross-channel inverse covariance
        # (shape_sens == (nc, nc)); AET/AE runs a per-channel diagonal.
        is_xyz = len(acs.shape_sens) == 2

        if is_wdm:
            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)
            Nf = int(bs.Nf_active)
            Nt = int(bs.Nt_active)
            k0 = np.clip(
                np.ceil(be[:-1] / layer_df - 1e-9).astype(int) - ind_min_f, 0, Nf
            )
            k1 = np.clip(
                np.floor(be[1:] / layer_df + 1e-9).astype(int) + 1 - ind_min_f,
                0, Nf,
            )
            # real WDM coefficients: 1 dof per (channel, layer, time) pixel
            dof_per_bin = nchannels * Nt
        else:
            df = float(bs.df)
            start_bin = int(acs.start_freq_ind[0])
            Nf = int(acs.data_length)
            k0 = np.clip(np.rint(be[:-1] / df).astype(int) - start_bin, 0, Nf)
            k1 = np.clip(np.rint(be[1:] / df).astype(int) - start_bin, 0, Nf)
            # complex FD bins: 2 real dof per (channel, bin)
            dof_per_bin = 2 * nchannels
        k1 = np.maximum(k1, k0)
        self._band_dof = (k1 - k0) * dof_per_bin

        def _shard_band_lls(r, ic):
            # r: (nw, nc, Nf[, Nt]); ic: (nw, nc[, nc], Nf[, Nt]) -> (nw, Nf)
            if is_xyz:
                if is_wdm:
                    per_bin = xp.einsum("wifk,wijfk,wjfk->wf", r, ic.real, r)
                else:
                    per_bin = xp.einsum(
                        "wif,wijf,wjf->wf", r.conj(), ic, r
                    ).real
            else:
                if is_wdm:
                    per_bin = xp.einsum("wifk,wifk,wifk->wf", r, ic.real, r)
                else:
                    per_bin = ((r.conj() * r).real * ic.real).sum(axis=1)
            cs = xp.zeros((per_bin.shape[0], Nf + 1))
            cs[:, 1:] = xp.cumsum(per_bin, axis=1)
            k0_xp, k1_xp = xp.asarray(k0), xp.asarray(k1)
            return -0.5 * norm * (cs[:, k1_xp] - cs[:, k0_xp])

        out = np.zeros((int(acs.acs_total_entries), num_bands))
        data_shaped, psd_shaped = acs.data_shaped, acs.psd_shaped
        for i, split in enumerate(acs.gpu_splits):
            if acs.gpus is not None:
                with xp.cuda.Device(acs.gpus[i]):
                    out[np.asarray(split)] = _to_numpy(
                        _shard_band_lls(data_shaped[i], psd_shaped[i])
                    )
            else:
                out[np.asarray(split)] = _to_numpy(
                    _shard_band_lls(data_shaped[i], psd_shaped[i])
                )
        return out

    def _update_band_leaf_caps(self, model, new_state, band_counts) -> None:
        """Advance the per-band progressive leaf caps (once per iteration).

        Runs at the very end of ``propose`` (after the final
        ``check_ll_inject`` rebuild, so the parent residual reflects the
        accepted state). Per band ``b``, the cap increments when ALL of:

        1. ``band_cap_iters[b] >= leaf_cap_min_iters`` at the current cap;
        2. every cold walker's band residual ll sits within
           ``leaf_cap_ll_nsigma * sqrt(N_b / 2)`` of the running best
           (``N_b`` = real dof in the band -- for a whitened residual
           ``-1/2<r|r>`` fluctuates with sigma ~ sqrt(N_b/2), so this is the
           "converged up to statistical change" test);
        3. (optional, ``leaf_cap_require_occupancy``) at least one cold
           walker actually holds ``cap[b]`` leaves in the band -- an
           exhausted band with free headroom keeps its cap.

        Bands increment independently; nothing waits on other bands. On
        increment the iteration counter and running best reset, so the next
        level must re-converge on its own evidence.
        """
        bi = new_state.sub_states[self.branch_name].band_info
        cap = bi["band_leaf_cap"]
        iters = bi["band_cap_iters"]
        best = bi["band_best_ll"]

        lls = self._band_residual_lls(model.analysis_container_arr)
        best[:] = np.maximum(best, lls.max(axis=0))
        iters += 1

        tol = self.leaf_cap_ll_nsigma * np.sqrt(self._band_dof / 2.0)
        converged = (iters >= self.leaf_cap_min_iters) & (
            (best - lls.min(axis=0)) <= tol
        )
        if self.leaf_cap_require_occupancy:
            cold_counts = _to_numpy(band_counts[0])  # (nwalkers, num_bands)
            converged &= cold_counts.max(axis=0) >= cap
        nleaves_max = self._work_branch(new_state).shape[2]
        converged &= cap < nleaves_max

        if np.any(converged):
            inc_bands = np.where(converged)[0]
            cap[converged] += 1
            iters[converged] = 0
            best[converged] = -np.inf
            logger.info(
                f"{self.name}: leaf cap incremented for bands "
                f"{inc_bands.tolist()} -> {cap[inc_bands].tolist()}."
            )
        logger.info(
            f"{self.name}: leaf caps min/max = {int(cap.min())}/{int(cap.max())}; "
            f"bands at min-iters gate: {int((iters < self.leaf_cap_min_iters).sum())}."
        )

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`GFState`): Current state of the sampler.

        Returns:
            :class:`GFState`: GFState of sampler after proposal is complete.

        """

        st_all = time.perf_counter()

        # Per-propose stage timing (GPU-efficiency diagnosis): one INFO line
        # per propose with the sorted stage breakdown. GB_PROP_TIMING_SYNC=1
        # synchronizes the device at every span boundary so device work is
        # attributed to the launching stage (see _ProposeTimer docstring).
        _tm_sync = None
        if self.backend.uses_cupy and os.environ.get("GB_PROP_TIMING_SYNC", "0") == "1":
            _tm_sync = self.xp.cuda.runtime.deviceSynchronize
        self._prop_timer = tm = _ProposeTimer(sync_fn=_tm_sync)
        # Propose-scoped SubBandBuffer cache: one allocation per signature
        # for the whole proposal (units rebind in place); torn down with a
        # memory-checker summary right before the final return.
        self._prop_buffer_cache = {}
        self._prop_buffer_builds = 0

        pin_main_device(self.xp, model.analysis_container_arr.gpus)
        # Run-time source of truth is the ACA that arrives with the model:
        # refresh the domain quantities and re-bind the parent engine (FD
        # prototype comp + move-level engine) if this ACA differs from the
        # one currently bound. All fills / likelihoods below go through
        # model.analysis_container_arr.
        self._configure_domain(model.analysis_container_arr)
        self._bind_parent_acs(model.analysis_container_arr)
        self.current_state = state
        # np.random.seed(10)
        # print("start stretch")

        # cold-row agreement between the main state and this branch's
        # sub-state (GF_SUBSTATE_CHECK=0 disables)
        self._check_substate_consistency(state, [self.branch_name])

        # The working ensemble is the SUB-STATE's tempered branch (module
        # ladder); the main state carries only the engine's cold chain, so
        # eryn-facing ``accepted`` arrays use the engine shape.
        work_in = self._work_branch(state)
        ntemps, nwalkers, nleaves_max, ndim = work_in.coords.shape
        engine_ntemps = state.log_like.shape[0]

        if not self.is_rj_prop and not np.any(work_in.inds):
            return state, np.zeros((engine_ntemps, nwalkers), dtype=bool)

        self.nwalkers = nwalkers
        self.ntemps = ntemps

        # Arm the per-band progressive leaf cap (search mode). The cap array
        # lives in ``state.band_info`` (HDF5-persisted); a fresh state
        # carries the -1 sentinel and is armed to ``leaf_cap_start`` here.
        # ``self._band_leaf_cap`` is a live numpy reference: the birth gate
        # in ``_run_rj_step`` reads it and ``_update_band_leaf_caps``
        # mutates it in place.
        self._band_leaf_cap = None
        if self._leaf_cap_enabled and self.is_rj_prop:
            bi = state.sub_states[self.branch_name].band_info
            ensure_leaf_cap_fields(bi, self.num_bands)
            if np.all(bi["band_leaf_cap"] < 0):
                bi["band_leaf_cap"][:] = int(self.leaf_cap_start)
                logger.info(
                    f"{self.name}: armed per-band leaf cap at "
                    f"{int(self.leaf_cap_start)} for {self.num_bands} bands."
                )
            self._band_leaf_cap = bi["band_leaf_cap"]

        # Run any move-specific setup.
        self.setup(model, state.branches)
        self.num_proposals += 1

        # An RJ move without a proposal distribution (e.g. search/refit
        # variants whose setup() has not produced one yet) cannot run. Pure
        # in-model moves don't need one.
        if self.is_rj_prop and self.rj_proposal_distribution is None:
            return state, np.zeros((engine_ntemps, nwalkers), dtype=bool)

        new_state = GFState(state, copy=True)
        assert new_state.log_like is not None

        # the copy's working branch: a view over new_state's sub-state arrays
        work = self._work_branch(new_state)

        band_temps = cp.asarray(state.sub_states[self.branch_name].band_info["band_temps"].copy())

        if self.is_rj_prop:
            orig_store = new_state.log_like[0].copy()

        gb_coords = cp.asarray(work.coords)

        self.mempool.free_all_blocks()

        waveform_kwargs_now = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs_now:
            waveform_kwargs_now.pop("N")

        rj_prop = None if not self.is_rj_prop else self.rj_proposal_distribution[self.branch_name]

        # make sure all periodic parameters have been put into their range
        work.coords[:] = self.periodic.wrap(
            {self.branch_name: work.coords[:].reshape(ntemps * nwalkers, nleaves_max, ndim)}
        )[self.branch_name].reshape(ntemps, nwalkers, nleaves_max, ndim)

        # TODO Ask Michael about this print("is this okay for rj? I do not think so, check with below use of gb_inds_in")
        if self.use_prior_removal:  # TODO: make this stronger?
            keep_all_inds = False
        else:
            keep_all_inds = True

        with tm.span("sorter_build"):
            self._sorter_dh = None
            self._sorter_hh = None
            band_sorter = BandSorter(
                work,
                self.band_edges,
                self.band_N_vals,
                force_backend=self.force_backend,
                transform_fn=self.parameter_transforms,
                max_data_store_size=self.max_data_store_size,
                gb=self.gb,
                gb_wdm_comp=self.gb_wdm_comp,
                gb_fd_comp=self.gb_fd_comp,
                wdm_band_slab_layers=self.wdm_band_slab_layers,
                wdm_slab_guard_layers=self.wdm_slab_guard_layers,
                waveform_kwargs=self.waveform_kwargs,
                rj_prop=rj_prop,
                keep_all_inds=keep_all_inds,
            )

        # Cold-chain friend table for the group-stretch half of the in-model
        # mix (rebuilt every proposal; cheap sort of the cold-chain f0s).
        self._infomat_wdm_logged = False
        if self.stretch_probability > 0.0 and self._build_friend_table:
            with tm.span("friend_index"):
                band_sorter.build_friend_index(self.nfriends)

        do_synchronize = False
        device = self.xp.cuda.runtime.getDevice() if self.backend.uses_cupy else -1

        # get non-gb contribution
        with tm.span("resid_open_close"):
            self.remove_cold_chain_sources_from_residual(model, band_sorter, apply_inds=True)
            # Multi-GPU: snapshot every per-GPU shard of linear_data_arr inside its
            # owning device context so the copies live on the right device. Restored
            # in check_ll_inject() symmetrically.
            self.reset_non_gb_linear_data_arr = self._snapshot_linear_data_arr(
                model.analysis_container_arr
            )
            self.add_cold_chain_sources_to_residual(model, band_sorter, apply_inds=True)
        # NOTE: no explicit source_only here — follow the run-level container
        # default so this baseline stays in the same convention as the
        # incremental checks below and check_ll_inject().
        with tm.span("ll_checks"):
            ll_after = model.analysis_container_arr.likelihood()  #  - cp.sum(cp.log(cp.asarray(psd[:2])), axis=(0, 2))).get()

        # print(np.abs(new_state.log_like - ll_after).max())        
        # store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        start_diffs = np.abs(new_state.log_like[0] - ll_after)

        check = ll_after - new_state.log_like[0] - start_diffs

        logger.debug(f"Start check: {start_diffs=}, {check=}")
        if not np.abs(check).max() < 1e-4:
            # assert np.abs(check).max() < 1.0
            new_state.log_like[0] = self.check_ll_inject(model, band_sorter)
            #? update start diffs
            start_diffs = np.abs(new_state.log_like[0] - ll_after)

        # print("CHECKING 0:", store_max_diff, self.is_rj_prop)
        # self.check_ll_inject(new_state, verbose=True)
        # assert np.all(start_diffs < 2.0)
        num_active_leaves = work.inds[0].sum(axis=-1) # cold chain only
        logger.info(f"Number of active leaves before proposal: {num_active_leaves}")
        # TODO: make sure band temps transfers out
        st_prop = time.perf_counter()
        with tm.span("run_proposal"):
            ll_change_log, prop_counts, acc_counts = self.run_proposal(
                model, new_state, band_sorter, band_temps
            )
        et_prop = time.perf_counter()
        # Diagnostic: per-temperature alive source counts after run_proposal
        _alive_per_temp_post_prop = [
            int(band_sorter.inds[band_sorter.temp_inds == _t].sum()) for _t in range(ntemps)
        ]
        logger.info(f"Alive sources per temp after run_proposal: {_alive_per_temp_post_prop}")
        logger.info(f"Runtime of {self.name} proposal is {round(et_prop - st_prop,3)} seconds.")

        # TODO ask michael about this print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum = ll_change_log.sum(axis=-1)
        new_state.log_like[0] += _to_numpy(ll_change_sum[0])

        with tm.span("ll_checks"):
            ll_after = model.analysis_container_arr.likelihood()
        check = ll_after - new_state.log_like[0] - start_diffs

        logger.debug(f"After proposal check: {start_diffs=}, {check=}")
        drift = float(np.abs(check).max())
        if drift >= 1e-4:
            # Incremental per-accept bookkeeping drifted from the true
            # residual likelihood (the narrow-band inner product reads a few
            # layers beyond the central band, whose context differs between
            # the cell buffer and the closed parent). The rebuild below is
            # exact, so the sampler stays correct; the warning tracks how
            # large the incremental drift got.
            logger.warning(
                f"{self.name}: incremental ll drift {drift:.3e} after "
                "proposal; rebuilding log_like from the residual."
            )
            with tm.span("ll_inject_drift"):
                new_state.log_like[0] = self.check_ll_inject(model, band_sorter)
        # breakpoint()

        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        # TODO: move this and check if it is needed
        # self.nchannels = model.analysis_container_arr.nchannels

        band_swaps_accepted = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)

        if (
            self.temperature_control is not None
            and self.time % 1 == 0
            and self.ntemps > 1
            and (self.is_rj_prop or self.swap_on_in_model)
            and self.run_swaps
            # and False
        ):
            st_temp = time.perf_counter()
            with tm.span("ll_checks"):
                ll_before1 = model.analysis_container_arr.likelihood()

            with tm.span("run_tempering"):
                ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed = self.run_tempering(
                    model, new_state, band_sorter, band_temps
                )

            new_state.log_like[0] += _to_numpy(ll_change_sum_temp[0])

            with tm.span("ll_checks"):
                ll_after = model.analysis_container_arr.likelihood()
            check = ll_after - new_state.log_like[0] - start_diffs

            logger.debug(f"After tempering check: {start_diffs=}, {check=}")
            drift = float(np.abs(check).max())
            if drift >= 1e-4:
                logger.warning(
                    f"{self.name}: incremental ll drift {drift:.3e} after "
                    "tempering; rebuilding log_like from the residual."
                )
                with tm.span("ll_inject_drift"):
                    new_state.log_like[0] = self.check_ll_inject(model, band_sorter)

            with tm.span("mempool_free"):
                self.mempool.free_all_blocks()
            et_temp = time.perf_counter()
            logger.info(f"Runtime of {self.name} tempering is {round(et_temp - st_temp,3)} seconds.")
            # Diagnostic: per-temperature alive source counts after run_tempering
            # _alive_per_temp_post_temp = [
            #     int(band_sorter.inds[band_sorter.temp_inds == _t].sum()) for _t in range(ntemps)
            # ]
            # logger.info(f"Alive sources per temp after run_tempering: {_alive_per_temp_post_temp}")

        # TODO ask michael about this print("make sure this works for rj")
        with tm.span("write_back"):
            self._write_back_state(new_state, band_sorter)

        et_all = time.perf_counter()
        logger.info(f"Full runtime of {self.name} is {round(et_all - st_all, 3)} seconds.")
        num_active_leaves = work.inds[0].sum(axis=-1)
        logger.info(f"Number of active leaves in cold chain after proposal: {num_active_leaves}")

        new_inds = cp.asarray(work.inds)
        del band_sorter
        with tm.span("mempool_free"):
            self.mempool.free_all_blocks()
        with tm.span("sorter_rebuild"):
            new_band_sorter = BandSorter(
                work,
                self.band_edges,
                self.band_N_vals,
                force_backend=self.force_backend,
                transform_fn=self.parameter_transforms,
                max_data_store_size=self.max_data_store_size,
                gb=self.gb,
                gb_wdm_comp=self.gb_wdm_comp,
                gb_fd_comp=self.gb_fd_comp,
                wdm_band_slab_layers=self.wdm_band_slab_layers,
                wdm_slab_guard_layers=self.wdm_slab_guard_layers,
                waveform_kwargs=self.waveform_kwargs,
            )

        # in-model inds will not change
        tmp_freqs_find_bands = cp.asarray(work.coords[:, :, :, 1])

        # calculate current band counts
        band_here = (
            cp.searchsorted(self.band_edges, tmp_freqs_find_bands.flatten() / 1e3, side="right") - 1
        ).reshape(tmp_freqs_find_bands.shape)

        group_temp_finder = [
            cp.repeat(cp.arange(ntemps), nwalkers * nleaves_max).reshape(
                ntemps, nwalkers, nleaves_max
            ),
            cp.tile(cp.arange(nwalkers), (ntemps, nleaves_max, 1)).transpose((0, 2, 1)),
            cp.tile(cp.arange(nleaves_max), ((ntemps, nwalkers, 1))),
        ]

        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        with tm.span("mempool_free"):
            self.mempool.free_all_blocks()

        self.time += 1
        # self.xp.cuda.runtime.deviceSynchronize()

        with tm.span("band_info"):
            band_info = new_band_sorter.get_band_info()

        # prop/acc counts: row 0 = RJ, row 1 = in-model; band_info wants
        # (num_bands, ntemps) summed over walkers. The two families are
        # recorded separately (one propose produces both kinds).
        sub = new_state.sub_states[self.branch_name]
        sub.band_info["band_temps"][:] = _to_numpy(band_temps)
        sub.band_info["band_num_binaries"][:] = band_info["band_counts"]
        sub.accumulate_proposals(
            _to_numpy(prop_counts[0].sum(axis=1).T),
            _to_numpy(acc_counts[0].sum(axis=1).T),
            is_rj=True,
        )
        sub.accumulate_proposals(
            _to_numpy(prop_counts[1].sum(axis=1).T),
            _to_numpy(acc_counts[1].sum(axis=1).T),
            is_rj=False,
        )
        sub.accumulate_swaps(
            _to_numpy(band_swaps_proposed), _to_numpy(band_swaps_accepted)
        )
        # TODO: check rj numbers

        # new_state.log_like[:] = self.check_ll_inject(new_state)

        with tm.span("mempool_free"):
            self.mempool.free_all_blocks()
        with tm.span("ll_inject_final"):
            new_state.log_like[:] = self.check_ll_inject(model, new_band_sorter)

        # Per-band progressive leaf caps advance AFTER the final residual
        # rebuild so the convergence metric sees the accepted state. Only
        # the designated updater move (one RJ move per iteration) advances
        # the counters; every cap-enabled RJ move enforces the gate.
        if self._band_leaf_cap is not None and self.leaf_cap_update:
            self._update_band_leaf_caps(model, new_state, band_info["band_counts"])

        # if self.is_rj_prop:
        #     pass  # print(self.name, "2nd count check:", new_state.branches[self.branch_name].inds.sum(axis=-1).mean(axis=-1), "\nll:", new_state.log_like[0] - orig_store, new_state.log_like[0])

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplemental)
        accepted = np.zeros((engine_ntemps, nwalkers), dtype=bool)

        num_active_sources = work.inds.sum(axis=-1)[0]
        logger.info(f"Current number of active sources in cold chain is {num_active_sources}")

        # Stage-timing breakdown for this propose (see _ProposeTimer).
        logger.info(
            "[GB_TIMING %s] %s",
            self.name,
            tm.report(time.perf_counter() - st_all),
        )

        self._buffer_cache_teardown()

        return new_state, accepted

    def check_ll_inject(self, model, band_sorter, verbose=False):
        # breakpoint()
        init_like = model.analysis_container_arr.likelihood()
        model.analysis_container_arr.zero_out_data_arr()
        # Restore the non-GB residual snapshot per-shard inside each owning
        # device context. Matches the snapshot loop in propose().
        self._restore_linear_data_arr(
            model.analysis_container_arr, self.reset_non_gb_linear_data_arr
        )
        self.add_cold_chain_sources_to_residual(model, band_sorter, apply_inds=True)
        final_like = model.analysis_container_arr.likelihood()
        return final_like

    @staticmethod
    def _snapshot_linear_data_arr(aca):
        """Thin alias for :meth:`AnalysisContainerArray.snapshot_linear_data_arr`."""
        return aca.snapshot_linear_data_arr()

    @staticmethod
    def _restore_linear_data_arr(aca, snapshot):
        """Thin alias for :meth:`AnalysisContainerArray.restore_linear_data_arr`."""
        aca.restore_linear_data_arr(snapshot)


class GBSpecialStretchMove(GBSpecialBase):
    """In-model GB move with the band-aware group-stretch / info-matrix mix.

    All machinery lives in :class:`GBSpecialBase`; the cold-chain friend
    table for the group stretch is rebuilt at the top of every ``propose``
    call (see ``build_friend_index``), so no per-iteration setup is needed
    here.
    """


class VGBSpecialStretchMove(GBSpecialBase):
    """In-model move for known (verification) galactic binaries.

    Fixed-dimensional (``nleaves_min == nleaves_max``, leaf i = one specific
    physical source at every walker/temperature), NO RJ. The proposal is a
    plain Goodman-Weare affine-invariant stretch over the sampled columns:
    each picked source is stretched against a random OTHER walker of the
    SAME physical source (same leaf, same temperature). No friend table, no
    info-matrix Cholesky, no phase maximization.

    The move stores nothing that feeds the proposal: it hands the CURRENT
    ensemble to stock :meth:`eryn.moves.StretchMove.get_proposal`, which does
    the affine-invariant sampling itself (picks each mover's complement,
    draws the stretch factor ``z``, and returns the ``(ndim - 1) * log(z)``
    detailed-balance factors — the coords are natively the reduced sampled
    basis, so nothing to adjust). ``sequential_parity_repeats = True`` makes
    the base repeat block run each repeat as eryn's red-blue split: even-
    parity walkers move against the current odd half, then odd against the
    UPDATED even half, with ``band_sorter.coords`` synced before each half.
    Every half-sweep is an invariant kernel, so ``num_repeat_proposals``
    (``VGB_NUM_REPEAT_PROPOSALS``) is a cost knob, not a bias knob.

    The fixed per-leaf parameters (f0, sky) live in the transform
    container's per-leaf ``fill_dict`` (Eryn per-leaf fills, selected by
    ``leaf_inds``).
    """

    # This move IS eryn's plain stretch. GBSpecialBase's MRO inherits
    # GroupStretchMove(GroupMove, StretchMove), and GroupMove overrides
    # choose_c_vals with a friend-table variant of a DIFFERENT signature --
    # re-point the pieces StretchMove.get_proposal dispatches through so the
    # stock proposal runs the textbook affine-invariant sweep. (The VGB flow
    # never builds a friend table: _build_friend_table = False below.)
    choose_c_vals = StretchMove.choose_c_vals
    get_new_points = StretchMove.get_new_points

    # Base repeat block runs eryn's red-blue split per repeat (see
    # GBSpecialBase._run_in_model_repeats): repeats compose invariant
    # kernels, so the repeat count is free.
    sequential_parity_repeats = True

    def __init__(self, *args, **kwargs):
        if kwargs.get("rj_proposal_distribution") is not None or kwargs.get(
            "is_rj_prop"
        ):
            raise ValueError("VGBSpecialStretchMove is in-model only (no RJ).")
        if kwargs.get("phase_maximize"):
            raise ValueError("Phase maximization is not used for VGBs.")
        kwargs.setdefault("branch_name", "vgb")
        kwargs.setdefault("use_info_mat_proposal", False)
        # No RJ move carries the band-temperature swaps for this branch.
        kwargs.setdefault("swap_on_in_model", True)
        kwargs.setdefault("stretch_probability", 1.0)
        super().__init__(*args, **kwargs)
        # Eryn's stretch picks the complement itself; no friend table needed.
        self._build_friend_table = False

    def in_model_proposal(self, coords, chol, band_sorter, source_ids, model):
        """One parity HALF of a Goodman-Weare red-blue sweep, via eryn's stretch.

        The base repeat block (``sequential_parity_repeats = True``) calls
        this once per parity half per repeat -- even-parity movers first,
        then odd -- and syncs ``band_sorter.coords`` to the tracked
        coordinates before each call, so the complement read here is the
        CURRENT opposite half (for the second half, including this repeat's
        accepted moves). That is eryn's :class:`~eryn.moves.RedBlueMove`
        split structure; each half-sweep is an invariant kernel.

        The proposal itself is stock
        :meth:`eryn.moves.StretchMove.get_proposal` -- complement draw,
        affine stretch, ``(ndim - 1) * log(z)`` factors -- with each mover
        entering as its own single-walker row and its opposite-parity
        walkers of the SAME ``(temperature, leaf)`` as the complement pool.
        The class-level ``choose_c_vals`` / ``get_new_points`` aliases keep
        ``get_proposal``'s internal dispatch on the plain stretch rather
        than GroupMove's friend-table overrides.
        """
        xp = self.xp
        nw = self.nwalkers
        assert nw >= 2 and nw % 2 == 0, (
            "the VGB red-blue stretch needs an even walker count >= 2"
        )
        t_i = band_sorter.temp_inds[source_ids]
        w_i = band_sorter.walker_inds[source_ids]
        l_i = band_sorter.leaf_inds[source_ids]
        ndim = coords.shape[-1]

        # CURRENT ensemble as (ntemps, nwalkers, nleaves, ndim) -- the base
        # repeat block wrote ``curr`` back just before this call. Valid
        # because VGB is fixed-dimensional: every leaf is alive at every
        # (temp, walker), so the flattened sorter coords reshape to the grid.
        ens4 = band_sorter.coords.reshape(
            self.ntemps, nw, band_sorter.nleaves_max, ndim
        )
        # complement pool per mover: the walkers of its (temp, leaf) in the
        # OPPOSITE parity (the other half of the red-blue split).
        walker_axis = xp.arange(nw)
        even_ws = walker_axis[walker_axis % 2 == 0]
        odd_ws = walker_axis[walker_axis % 2 == 1]
        # (n_src, nw/2) opposite-parity walker indices for each mover
        opp_ws = xp.where((w_i % 2 == 0)[:, None], odd_ws[None, :], even_ws[None, :])
        comp = ens4[t_i[:, None], opp_ws, l_i[:, None], :]        # (n_src, nw/2, ndim)

        # Stock eryn stretch: each mover is its own single-walker row on the
        # leading axis with its own complement pool.
        rng = model.random if not self.use_gpu else self.xp.random
        newpos, factors = StretchMove.get_proposal(
            self,
            {self.branch_name: coords[:, None, None, :]},        # (n, 1, 1, ndim)
            {self.branch_name: [comp[:, :, None, :]]},           # (n, nc, 1, ndim)
            rng,
        )
        return newpos[self.branch_name][:, 0, 0, :], factors.reshape(-1)


class GBSpecialRJPriorMove(GBSpecialBase):
    """Reversible-jump GB move that draws proposals from the prior distribution."""
    pass


def para_log_like(
    x,
    gb,
    acs,
    walker_max,
    transform_fn,
    phase_maximize,
    waveform_kwargs,
    fstat=True,
    return_snr=False,
):
    """Vectorized GB log-likelihood used by serial-search and refit moves.

    Args:
        x: GB parameter rows (untransformed).
        gb: :class:`gbgpu.GBGPU` instance.
        acs: :class:`AnalysisContainerArray`.
        walker_max: Index of the walker whose data the proposals are scored
            against.
        transform_fn: :class:`TransformContainer` for GB parameters.
        phase_maximize: If ``True``, marginalize over phase.
        waveform_kwargs: Forwarded to ``gb.get_fstat_ll`` / ``gb.get_ll``.
        fstat: If ``True``, use the F-statistic likelihood (``get_fstat_ll``)
            and overwrite the amplitude / phase / iota / polarization
            entries of ``x`` with their maximized values.
        return_snr: If ``True`` and ``fstat`` is ``False``, also return the
            optimal SNR per row.

    Returns:
        Per-row log-likelihood (or ``(ll, snr)`` tuple).
    """
    xp = gb.backend.xp

    x_tmp = transform_fn.both_transforms(x, xp=xp)
    # need to get just f, fdot, fddot, alpha, delta
    data_index = xp.full(x.shape[0], walker_max, dtype=xp.int32)
    if fstat:
        x_in = x_tmp[:, xp.array([1, 2, 3, 7, 8])]
        # breakpoint()
        # TODO: fix for N>256?
        ll = gb.get_fstat_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.end_shape[0],
            data_splits=np.array([gb.gpus[0]]),
            phase_maximize=phase_maximize,
            return_cupy=True,
            N=512,  
            **waveform_kwargs,
        )

        # TODO(gb fstat-dist-birth step 2): examine exactly what this
        # post-get_fstat_ll writeback does and WHY it lives here. It pins the
        # sampling coords to the 4-parameter F-stat maximum -- amplitude
        # (A_max, converted A_max -> distance for the distance basis),
        # phi0_max, iota_max, psi_max -- computed by gbgpu.get_fstat_ll's
        # Jaranowski-Krol inversion (a_i = M^-1 N -> A_plus/A_cross -> A_max;
        # gbgpu.py:1370-1391). The RJ birth path must REUSE this A_max ->
        # distance center (NOT the empirical d_h/h_h residual rescale added in
        # step 1) and then draw the birth distance from a distribution about
        # that center (ln dist ~ N(ln dist*, 1/SNR*), SNR*^2 = 2F), with the
        # proposal density carried into the RJ factor. Decide here whether the
        # deterministic pin of phi0/iota/psi is what we want for the birth
        # proposal or whether those also need a spread for detailed balance.
        #
        # Write the F-stat-maximized (amplitude, phi0, iota, psi) back into
        # the sampling coords. Slot 0 is lnA in the amplitude basis, but the
        # DISTANCE basis samples distance there -- convert A_max -> distance
        # (A propto 1/d: dist = gb_amp_from_dist(f0, Mc, 1) / A_max) or the
        # maximized amplitude lands out of the distance prior box and every
        # birth is rejected.
        if list(getattr(transform_fn, "input_basis", []))[:1] == ["dist"]:
            from ..stock.erebor.transforms import gb_amp_from_dist

            x[:, 0] = gb_amp_from_dist(x[:, 1] * 1e-3, x[:, 2], 1.0) / gb.A_max
        else:
            x[:, 0] = np.log(gb.A_max)
        x[:, 3] = gb.phi0_max % (2 * np.pi)
        x[:, 4] = np.cos(gb.iota_max % (np.pi))
        x[:, 5] = gb.psi_max % (np.pi)

    else:
        # breakpoint()
        x_in = x_tmp[:]
        ll = gb.get_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.end_shape[0],
            data_splits=np.array([gb.gpus[0]]),
            phase_maximize=phase_maximize,
            return_cupy=True,
            # N=512,
            **waveform_kwargs,
        )
        # breakpoint()

        # params_remove_in = x_in.copy()
        # params_add_in = x_in.copy()

        # params_remove_in[:, 0] *= 1e-50
        # breakpoint()
        # ll_diff_2 = gb.swap_likelihood_difference(
        #     params_remove_in,
        #     params_add_in,
        #     acs.linear_data_arr,
        #     acs.linear_psd_arr,
        #     # start_freq_ind=self.xp.asarray(self.acs.start_freq_ind).astype(np.int32),
        #     data_index=data_index,
        #     noise_index=data_index,
        #     # N=N_vals,
        #     data_length=acs.data_length,
        #     data_splits=np.array([gb.gpus[0]]),
        #     phase_maximize=phase_maximize,
        #     return_cupy=True,
        #     N=256,
        #     **waveform_kwargs,
        # )
        # breakpoint()

        if phase_maximize:
            x[:, 3] = (x[:, 3] - xp.angle(xp.asarray(gb.non_marg_d_h))) % (2 * np.pi)

        if return_snr:
            opt_snr = gb.h_h.real ** (1 / 2)
            return (ll, opt_snr)

    return ll


class PriorTransformFn:
    """Transform between unit-cube prior coordinates and GB :math:`(f, \\dot f)`.

    Used by the serial-search move to draw uniform-in-band proposals while
    keeping the rest of the GB prior unchanged.

    Args:
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz).
        fdot_min: Minimum frequency derivative.
        fdot_max: Maximum frequency derivative.
    """

    def __init__(self, f_min: float, f_max: float, fdot_min: float, fdot_max: float):
        self.f_min, self.f_max, self.fdot_min, self.fdot_max = (
            f_min,
            f_max,
            fdot_min,
            fdot_max,
        )

    def adjust_logp(self, logp, groups_running):
        """Add the (uniform) ``f`` and ``fdot`` log-density to ``logp``."""
        xp = get_array_module(self.f_min)

        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        f_logpdf = np.log(1.0 / (f_max_here - f_min_here))

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        fdot_logpdf = np.log(1.0 / (fdot_max_here - fdot_min_here))

        logp[:] += f_logpdf[:, None, None]
        logp[:] += fdot_logpdf[:, None, None]

        return logp

    def transform_to_prior_basis(self, coords, groups_running):
        """Map ``f`` / ``fdot`` columns of ``coords`` to the unit-cube basis."""
        xp = get_array_module(self.f_min)

        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        try:
            coords[:, :, :, 1] = (coords[:, :, :, 1] - f_min_here[:, None, None]) / (
                f_max_here[:, None, None] - f_min_here[:, None, None]
            )
        except:
            breakpoint()

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (coords[:, :, :, 2] - fdot_min_here[:, None, None]) / (
            fdot_max_here[:, None, None] - fdot_min_here[:, None, None]
        )

        return

    def transform_from_prior_basis(self, coords, groups_running):
        """Map ``f`` / ``fdot`` columns of ``coords`` from unit cube back to physical."""
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        assert groups_running.shape[0] == coords.shape[0]
        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        coords[:, :, :, 1] = (
            coords[:, :, :, 1] * (f_max_here[:, None, None] - f_min_here[:, None, None])
        ) + f_min_here[:, None, None]

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (
            coords[:, :, :, 2] * (fdot_max_here[:, None, None] - fdot_min_here[:, None, None])
        ) + fdot_min_here[:, None, None]

        return


def _gb_sampling_third_name(move) -> str:
    """Column-2 name of the run's GB sampling basis.

    ``"fdot"`` for the legacy basis, ``"Mc"`` for chirp-mass runs
    (``GBSettings.use_chirp_mass``), read off the move's transform-container
    ``input_basis`` -- so GMM RJ containers assemble in the SAME basis the
    sampler walks in.
    """
    ib = list(getattr(getattr(move, "transform_fn", None), "input_basis", None) or [])
    return "Mc" if "Mc" in ib else "fdot"


def _gb_fdot_astro_ratio_max(move):
    """Half-width M of the U[-M, M] fdot_astro_ratio prior, or None.

    Read off the move's own run prior so RJ-birth / GMM-refit containers
    append the 9th column with the SAME bounds the sampler walks in
    (``None`` for the 8-column bases). Independent of the move subclass.
    """
    if getattr(move, "_fdot_astro_col", None) is None:
        return None
    priors = getattr(move, "priors", None)
    cont = priors.get(move.branch_name) if isinstance(priors, dict) else None
    d = getattr(cont, "priors_in", {}).get("fdot_astro_ratio") if cont else None
    return float(d.maximum) if d is not None else None


def _gb_use_distance(move) -> bool:
    """True when slot 0 of the run's GB sampling basis is ``dist`` (kpc).

    Read off the move's transform-container ``input_basis`` so GMM/refit
    containers assemble slot 0 in the SAME basis the sampler walks in
    (``dist`` vs ``A``).
    """
    ib = list(getattr(getattr(move, "transform_fn", None), "input_basis", None) or [])
    return len(ib) > 0 and ib[0] == "dist"



from lisatools.sampling.gmm import fit_gb_gmm_rj_container

class GBSpecialRJSerialSearchMCMC(GBSpecialBase):
    """Reversible-jump GB move that runs a serial F-statistic MCMC search per band.

    Each band proposes one new GB at a time using a parallel ensemble
    sampler driven by :func:`para_log_like`, with proposals drawn from a
    band-restricted prior via :class:`PriorTransformFn`.
    """
    comm_info = None

    def _third_col_search_bounds(self, f0_max):
        """(min, max) arrays for the sampling-basis third column vs f0_max.

        Legacy fdot basis: the f-dependent ``get_fdot_mojito`` envelopes.
        Chirp-mass basis: the flat ``m_chirp_lims`` box (the F-stat search
        maximizes over the whole chirp-mass range) -- read off the run's Mc
        prior (GMM ``mc_lims`` or the uniform's bounds). Fixing the third
        column to the RIGHT quantity is the pre-existing chirp-mass search
        bug (fdot bounds were being written into the Mc slot).
        """
        if _gb_sampling_third_name(self) != "Mc":
            return get_fdot_mojito(f0_max, sign="-"), get_fdot_mojito(f0_max, sign="+")
        pr = self.priors[self.branch_name].priors_in
        gmm = pr.get(("f0", "Mc"))
        if gmm is not None and getattr(gmm, "mc_lims", None) is not None:
            lo, hi = gmm.mc_lims
        elif "Mc" in pr:
            lo, hi = float(pr["Mc"].minimum), float(pr["Mc"].maximum)
        else:
            lo, hi = 0.001, 1.0
        return np.full_like(f0_max, float(lo)), np.full_like(f0_max, float(hi))

    def _band_restricted_priors_in(self, priors_global):
        """priors_in with the f0 + third columns swapped for unit-cube uniforms.

        Preserves the run's column layout (string-key column = dict
        insertion order): the joint ``("f0", "Mc")`` GMM (or the legacy
        single ``f0``/``fdot`` keys) is replaced IN PLACE by two
        band-restricted unit-cube singles, so ``PriorTransformFn`` maps
        columns 1 and 2 back to physical. Every other prior -- crucially the
        9th ``fdot_astro_ratio`` U[-M, M] -- is carried through unchanged.
        """
        third = _gb_sampling_third_name(self)
        uc = self.backend.uses_cupy
        src = deepcopy(priors_global)[self.branch_name].priors_in
        out = {}
        for key, dist in src.items():
            if key in ("f0", "fdot", "Mc", ("f0", "Mc"), ("f0", "fdot")):
                # Emit the two band-restricted singles once, at the position
                # of the first f0/third key (keeps columns 1 and 2).
                if "f0" not in out:
                    out["f0"] = UniformDistribution(0.0, 1.0, use_cupy=uc)
                    out[third] = UniformDistribution(0.0, 1.0, use_cupy=uc)
            else:
                out[key] = dist
        return out

    def setup(self, model, branches):
        assert isinstance(self.search_kwargs, dict)
        nwalkers: int = self.search_kwargs["nwalkers"]
        ntemps: int = self.search_kwargs["ntemps"]
        shutoff_band_iteration: int = self.search_kwargs["shutoff_band_iteration"]
        shutoff_frequency_threshold: float = self.search_kwargs["shutoff_frequency_threshold"]
        burn_1: int = self.search_kwargs["burn_1"]
        nsteps_1: int = self.search_kwargs["nsteps_1"]
        snr_threshold: float = self.search_kwargs["snr_threshold"]
        burn_2: int = self.search_kwargs["burn_2"]
        nsteps_2: int = self.search_kwargs["nsteps_2"]

        # run paraensemble MCMC.
        max_logl_walker = np.argmax(model.analysis_container_arr.likelihood()).item()
        self.gb.d_d = model.analysis_container_arr.inner_product()[max_logl_walker] # 0.0
        ndim = branches[self.branch_name].ndim
        priors_global = self.priors if not self.backend.uses_cuda else self.gpu_priors            

        if self.num_bands == 1:
            f0_max = self.band_edges[1:]
            f0_min = self.band_edges[:-1]
        else:
            f0_max = self.band_edges[2:-1]
            f0_min = self.band_edges[1:-2]

        # logic to shutoff bands #? Think about how this should change when we change SNR_thresh and with changing noise
        if self.num_proposals >= shutoff_band_iteration:
            bands_to_shutoff = np.all(~self.found_source_in_band[-shutoff_band_iteration:, :], axis=0)

            if shutoff_frequency_threshold is not None:
                min_freqs = getattr(f0_min, "get")() if hasattr(f0_min, "get") else f0_min
                freq_mask = min_freqs >= shutoff_frequency_threshold
                bands_to_shutoff = bands_to_shutoff & freq_mask

            if np.all(bands_to_shutoff):
                logger.info(f"No sources found across all bands for {shutoff_band_iteration} iterations, reverting to priors")
                self.rj_proposal_distribution = priors_global
                return

            else:
                shutoff_mask = ~bands_to_shutoff
                f0_max = f0_max[shutoff_mask]
                f0_min = f0_min[shutoff_mask]
                ngroups = np.sum(shutoff_mask)

        else:
            ngroups = max(1, self.num_bands - 2)
            assert f0_max.shape[0] == ngroups
            assert f0_min.shape[0] == ngroups
            bands_to_shutoff = None

        logger.info(f"The current number of active bands is {ngroups}")

        # Band-restricted proposal bounds for the f0 + third columns. The
        # third column is fdot (legacy basis) or Mc (chirp-mass): its
        # PriorTransformFn range must match what it actually holds --
        # f-dependent fdot envelopes vs the flat chirp-mass box (the latter
        # was the pre-existing chirp-mass search bug). The 9th
        # fdot_astro_ratio column keeps its real U[-M, M] prior untouched
        # (drawn from prior, not band-remapped).
        third_min, third_max = self._third_col_search_bounds(f0_max)

        priors_in = self._band_restricted_priors_in(priors_global)
        _band_prior = ProbDistContainer(
            priors_in, return_gpu=True, use_cupy=self.backend.uses_cupy
        )
        if _gb_use_distance(self):
            # The ("dist","alpha","sin_delta") joint grabbed consecutive
            # columns by insertion order; remap by name to the real basis
            # (dist->0, alpha->6, sin_delta->7; f0/Mc singles at 1/2).
            _band_prior.reset_key_order(list(self.transform_fn.input_basis))
        priors = {self.branch_name: _band_prior}
        start_params = priors[self.branch_name].rvs(size=(ngroups, ntemps, nwalkers))
        prior_transform_fn = PriorTransformFn(f0_min * 1e3, f0_max * 1e3, third_min, third_max)
        prior_transform_fn.transform_from_prior_basis(start_params, self.xp.arange(ngroups))

        #? print("phase maximizing here right now (?)")
        ll_args = (
            self.gb,
            model.analysis_container_arr,
            max_logl_walker,
            self.parameter_transforms,
            True,  # self.phase_maximize,
            self.waveform_kwargs,
        )

        ll_args_2 = (
            self.gb,
            model.analysis_container_arr,
            max_logl_walker,
            self.parameter_transforms,
            self.phase_maximize, # False, #
            self.waveform_kwargs,
        )

        # test_ll = para_log_like(
        #     test_params,
        #     *ll_args
        # )

        # Stage 1 (F-stat MCMC): search f0 + intrinsic sky; A/phi0/cos_iota/
        # psi are analytically maximized (off). The fdot_astro_ratio column
        # (if present) is ALSO off -- the F-stat is exactly flat along the
        # (Mc, ratio) split, so it is held at its prior-drawn birth value
        # and refined by stage 2 / the in-model stretch.
        gibbs_sampling_setup = np.ones(ndim, dtype=bool)
        _off = [0, 3, 4, 5]
        if self._fdot_astro_col is not None:
            _off.append(self._fdot_astro_col)
        gibbs_sampling_setup[np.array(_off)] = False
        para_sampler = ParaEnsembleSampler(
            ndim,
            nwalkers,
            ngroups,
            para_log_like,
            priors,
            tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
            args=ll_args,
            # kwargs: dict = {},
            gpu=self.gb.gpus[0],
            periodic=self.periodic,
            # backend: ParaBackend = None,  # add ParaHDFBackend
            # update_fn: Callable = None,
            # update_iterations=-1,
            # stopping_fn: Callable = None,
            # stopping_iterations: int=-1,
            prior_transform_fn=prior_transform_fn,
            name=self.branch_name,
            gibbs_sampling_setup=gibbs_sampling_setup,
            # provide_supplemental=False,
        )

        from eryn.state import ParaState

        state = ParaState({self.branch_name: start_params}, groups_running=self.xp.ones(ngroups, dtype=bool))
        state.log_prior = para_sampler.compute_log_prior(state.branches_coords)
        state.log_like = para_sampler.compute_log_like(state.branches_coords, logp=state.log_prior)

        para_sampler.run_mcmc(state, nsteps_1, burn=burn_1, progress=getattr(self, "progress", False))

        samples = self.xp.asarray(para_sampler.get_chain()[:, :, 0])
        # Diagnostics disabled (stft_tof): computed-but-unused and each costs
        # a full likelihood sweep over all samples.
        # check_ll = para_sampler.get_log_like()[:, :, 0]
        # sample_ll = asnumpy(
        #     para_log_like(samples.reshape(-1, 8), *ll_args).reshape(samples.shape[:-1])
        # )

        # check_real_ll_phase_maximized = asnumpy(
        #     para_log_like(samples.reshape(-1, 8), *ll_args, fstat=False)
        #     .reshape(samples.shape[:-1])
        # )
        check_real_ll, opt_snr = para_log_like(
            samples.reshape(-1, samples.shape[-1]), *ll_args_2, fstat=False, return_snr=True
        )
        check_real_ll = asnumpy(check_real_ll.reshape(samples.shape[:-1]))
        opt_snr = asnumpy(opt_snr.reshape(samples.shape[:-1]))

        # np.save("opt_snr_from_ldc_parasampler_check.npy", opt_snr)
        # np.save("samples_from_ldc_parasampler_check.npy", samples)

        # TODO: make cut adjustable
        groups_running_now = opt_snr.min(axis=(0, 2)) > snr_threshold

        if self.num_proposals == 0:
            self.found_source_in_band = groups_running_now
        else:
            if bands_to_shutoff is None:
                self.found_source_in_band = np.vstack([self.found_source_in_band, groups_running_now])
            else:
                shutoff_temp = np.zeros(self.found_source_in_band.shape[1], dtype=bool)
                shutoff_temp[~bands_to_shutoff] = groups_running_now
                self.found_source_in_band = np.vstack([self.found_source_in_band, shutoff_temp])

        logger.info(f"Found a source in {groups_running_now.sum()} out of {groups_running_now.shape[0]} active bands")
        if not np.any(groups_running_now):
            logger.info("Did not find any new sources.")
            return

        start_params_2 = np.tile(samples[-1][groups_running_now, None], (1, ntemps, 1, 1))
        # Maybe not start from maximized values?
        # Stage 2 (full-likelihood refine): sample all columns INCLUDING
        # fdot_astro_ratio (it explores the degenerate (Mc, ratio) ridge
        # here); only phi0 drops out when phase-maximizing.
        gibbs_sampling_setup_2 = np.ones(ndim, dtype=bool)
        if ll_args_2[4]: # phase_maximization
            gibbs_sampling_setup_2[np.array([3])] = False

        prior_transform_fn_2 = PriorTransformFn(
            f0_min[groups_running_now] * 1e3,
            f0_max[groups_running_now] * 1e3,
            third_min[groups_running_now],
            third_max[groups_running_now],
        )
        ngroups_2 = groups_running_now.sum().item()
        # prior_transform_fn_2.transform_from_prior_basis(start_params_2, self.xp.arange(ngroups_2))

        para_sampler_2 = ParaEnsembleSampler(
            ndim,
            nwalkers,
            ngroups_2,
            para_log_like,
            priors,
            tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
            args=ll_args_2,
            kwargs=dict(fstat=False),
            gpu=self.gb.gpus[0],
            periodic=self.periodic,
            # backend: ParaBackend = None,  # add ParaHDFBackend
            # update_fn: Callable = None,
            # update_iterations=-1,
            # stopping_fn: Callable = None,
            # stopping_iterations: int=-1,
            prior_transform_fn=prior_transform_fn_2,
            name=self.branch_name,
            gibbs_sampling_setup=gibbs_sampling_setup_2,
            # provide_supplemental=False,
        )

        new_state = ParaState(
            {self.branch_name: start_params_2}, groups_running=self.xp.ones(ngroups_2, dtype=bool)
        )
        new_state.log_prior = para_sampler_2.compute_log_prior(new_state.branches_coords)
        new_state.log_like = para_sampler_2.compute_log_like(
            new_state.branches_coords, logp=new_state.log_prior
        )

        if np.any(np.isinf(new_state.log_prior)):
            breakpoint()

        para_sampler_2.run_mcmc(new_state, nsteps_2, burn=burn_2, progress=getattr(self, "progress", False))

        samples_2 = self.xp.asarray(para_sampler_2.get_chain()[:, :, 0])
        # check_ll_2 = para_sampler_2.get_log_like()[:, :, 0]

        # Diagnostics disabled (stft_tof): computed-but-unused likelihood sweeps.
        # check_real_ll_phase_maximized_2 = asnumpy(
        #     para_log_like(samples_2.reshape(-1, 8), *ll_args, fstat=False)
        #     .reshape(samples_2.shape[:-1])
        # )
        # check_real_ll_2 = asnumpy(
        #     para_log_like(samples_2.reshape(-1, 8), *ll_args_2, fstat=False)
        #     .reshape(samples_2.shape[:-1])
        # )

        samples_2 = samples_2.transpose(1, 0, 2, 3)
        # np.save("/workspace/rrondeel/erebor/testing/highf_gb/search2_samples_check.npy", samples_2)

        st = time.perf_counter()
        samples_2_tmp = samples_2.reshape(samples_2.shape[0], -1, samples_2.shape[-1])[
            :, :, np.array([0, 1, 2, 4, 6, 7])
        ]

        # Standalone samples -> batched-GMM-container entry point (guards +
        # vec_fit_gmm_min_bic + basis-aware container assembly live there).
        try:
            rj_dist = fit_gb_gmm_rj_container(
                samples_2_tmp,
                use_chirp_mass=_gb_sampling_third_name(self) == "Mc",
                use_cupy=True,
                gpu=self.xp.cuda.runtime.getDevice(),
                fdot_astro_ratio_max=_gb_fdot_astro_ratio_max(self),
                use_distance=_gb_use_distance(self),
            )
        except ValueError as e:
            logger.warning(f"GB search GMM fit skipped: {e}")
            return

        logger.info(
            f"Runtime of GPU GMM FIT: {round(time.perf_counter() - st, 3)} seconds"
        )
        self.rj_proposal_distribution = {self.branch_name: rj_dist}


# GBSpecialRJSearchMove (the MPI multi-GPU search delegate) was removed
# with the move->rank dispatch (parallel-resources plan P3); the serial
# single-GPU search (GBSpecialRJSerialSearchMCMC + GPU GMM fitting) is
# the live search path.


from lisatools.globalfit.gathergalaxy import gather_gb_samples
from lisatools.globalfit.hdfbackend import GBHDFBackend, GFHDFBackend, MBHHDFBackend
from lisatools.globalfit.state import GBState


class GBSpecialRJRefitMove(GBSpecialBase):
    """Reversible-jump GB move that uses GMM-refitted proposals.

    Loads per-leaf GMM proposals (refit by :func:`gb_refit_func`) and uses
    them as the RJ proposal distribution. This is typically alternated
    with :class:`GBSpecialRJPriorMove` to sharpen accepted GB candidates.
    """
    def __init__(self, *args, fp=None, **kwargs):
        assert fp is not None and isinstance(fp, str)
        assert os.path.exists(fp)
        self.fp = fp
        GBSpecialBase.__init__(self, *args, **kwargs)

    def setup(self, model, branches):
        samples_keep = self.search_kwargs["refit_start_iteration"]
        nwalkers = self.search_kwargs["nwalkers"]
        num_compare_samples = 1

        max_logl_walker = np.argmax(model.analysis_container_arr.likelihood()).item()
        self.gb.d_d = 0.0  # model.analysis_container_arr.inner_product()[max_logl_walker]
        reader = GFHDFBackend(
            self.fp, sub_state_bases={self.branch_name: GBState}, sub_backend={self.branch_name: GBHDFBackend}
        )

        st = time.perf_counter()
        sens_mat = model.analysis_container_arr[max_logl_walker].sens_mat
        if reader.iteration < 2 * samples_keep:
            logger.info("Not enough samples to perform refitting, reverting to priors.")
            self.rj_proposal_distribution = {self.branch_name: self.priors if not self.backend.uses_cuda else self.gpu_priors}
            return

        num_compare_samples = 1
        nwalkers = 30
        gpu = self.xp.cuda.runtime.getDevice() if self.backend.uses_cupy else -1
        # ``gather_gb_samples`` builds FD waveforms for the GMM refit;
        # the WDM equivalent has not been wired in yet.
        if not isinstance(model.analysis_container_arr.settings, FDSettings):
            raise NotImplementedError(
                "GBSpecialRJRefitMove currently requires the FD basis "
                "(GMM refit fits FD waveforms)."
            )
        fd = model.analysis_container_arr.f_arr.copy()
        groups = gather_gb_samples(
            fd,
            self.parameter_transforms,
            self.gb,
            self.waveform_kwargs.copy(),
            self.band_edges,
            self.band_N_vals,
            reader,
            sens_mat,
            gpu,
            num_compare_samples=num_compare_samples,
            samples_keep=samples_keep,
            thin_by=1,
        )

        num_in_groups = np.asarray([len(tmp) for tmp in groups])
        keep = num_in_groups > nwalkers * samples_keep / 2

        logger.info(
            f"Groups passing sample count filter: {keep.sum()} / {len(keep)}. "
            f"num_in_groups: {num_in_groups}"
        )

        if not keep.any():
            logger.warning(
                f"No groups have enough samples (threshold={nwalkers * samples_keep / 2:.0f}). "
                f"Max samples in any group: {num_in_groups.max()}. "
                f"Reverting to priors."
            )
            self.rj_proposal_distribution = {
                self.branch_name: self.priors if not self.backend.uses_cuda else self.gpu_priors
            }
            return

        max_num_source = max([tmp.shape[0] for tmp in groups])
        samples = np.full((len(groups), max_num_source, groups[0].shape[-1]), np.nan)
        for i, group in enumerate(groups):
            samples[i, : len(group)] = group

        samples_fin = samples[keep]
        num_in_groups_fin = num_in_groups[keep]

        if len(num_in_groups_fin) == 0 or num_in_groups_fin.min() == num_in_groups_fin.max():
            logger.warning(
                f"Cannot construct step range from num_in_groups_fin={num_in_groups_fin}. "
                f"Reverting to priors..."
            )
            self.rj_proposal_distribution = {
                self.branch_name: self.priors if not self.backend.uses_cuda else self.gpu_priors
            }
            return

        if self.backend.uses_cupy:
            self.xp.cuda.runtime.setDevice(gpu)
        output_info = []
        step = 5
        steps = np.arange(num_in_groups_fin.min(), num_in_groups_fin.max(), step)
        if steps[-1] < num_in_groups_fin.max().item():
            steps = np.concatenate([steps, np.array([num_in_groups_fin.max().item()])])

        weights_all = []
        means_all = []
        covs_all = []
        invcovs_all = []
        dets_all = []
        mins_all = []
        maxs_all = []
        for start, end in zip(steps[:-1], steps[1:]):
            here = (num_in_groups_fin >= start) & (num_in_groups_fin < end)
            # this randomly throughs away ~step amount of samples to make gmm work
            samples_here = samples_fin[here][:, :start, np.array([0, 1, 2, 4, 6, 7])].copy()

            if np.isnan(samples_here).any():
                nan_groups = np.where(np.isnan(samples_here).any(axis=(1, 2)))[0]
                logger.warning(
                    f"NaN padding leaked into samples_here at start={start}. \
                    Affected groups (local indices): {nan_groups}. \
                    num_in_groups for those groups: {num_in_groups_fin[here][nan_groups]} \
                    Skipping Refit..."
                )
                return
                # raise ValueError(
                #     f"NaN padding leaked into samples_here at start={start}. "
                #     f"Affected groups (local indices): {nan_groups}. "
                #     f"num_in_groups for those groups: {num_in_groups_fin[here][nan_groups]}"
                # )

            ranges = samples_here.max(axis=1) - samples_here.min(axis=1)
            if (ranges == 0).any():
                bad = np.where((ranges == 0))
                logger.warning(
                    f"Degenerate features at start={start}: groups={bad[0]}, features={bad[1]} \
                    Skipping Refit..."
                )
                # raise ValueError(
                #     f"Degenerate features at start={start}: groups={bad[0]}, features={bad[1]}"
                # )

            weights, means, covs, invcovs, dets, mins, maxs = vec_fit_gmm_min_bic(
                cp.asarray(samples_here),
                min_comp=1,
                max_comp=30,
                n_samp_bic_test=5000,
                gpu=gpu,
                verbose=False,
                return_components=True,
            )
            weights_all += weights
            means_all += means
            covs_all += covs
            invcovs_all += invcovs
            dets_all += dets
            mins_all += mins
            maxs_all += maxs
            # logger.info(start, end)

        full_gmm = FullGaussianMixtureModel(
            weights_all,
            means_all,
            covs_all,
            invcovs_all,
            dets_all,
            mins_all,
            maxs_all,
            use_cupy=True,
        )

        logger.info(f"Runtime GMM Refit: {round(time.perf_counter() - st)}")
        # Basis-aware container: column 2 is "fdot" (legacy) or "Mc"
        # (use_chirp_mass runs) -- the refitted chains are in the run's
        # sampling basis, so the container's key map must follow it.
        _third = _gb_sampling_third_name(self)
        # slot 0 is "dist" (distance basis) or "A" (lnA), read off the run basis.
        _first = "dist" if _gb_use_distance(self) else "A"
        _refit_priors = {
            (_first, "f0", _third, "cos_iota", "alpha", "sin_delta"): full_gmm,
            "phi0": UniformDistribution(0.0, 2 * np.pi),
            "psi": UniformDistribution(0.0, np.pi),
        }
        _key_order = [_first, "f0", _third, "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
        _ratio_max = _gb_fdot_astro_ratio_max(self)
        if _ratio_max is not None:
            # 9th column: refit births draw the fdot_astro ratio from its
            # U[-M, M] prior (the GMM fit only the 6 intrinsic columns).
            _refit_priors["fdot_astro_ratio"] = UniformDistribution(
                -_ratio_max, _ratio_max
            )
            _key_order.append("fdot_astro_ratio")
        rj_dist = ProbDistContainer(_refit_priors, use_cupy=True)
        rj_dist.reset_key_order(_key_order)
        self.rj_proposal_distribution = {self.branch_name: rj_dist}


def get_param_limits(array): # can be used for debugging of coordinate values
    """Return per-column min/max of ``array`` (debug helper for GB coordinates)."""
    num_params = array.shape[-1]

    if num_params == 8:
        param_labels = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
    elif num_params == 9:
        param_labels = ["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
    else:
        param_labels = num_params * [""]
    for i, param_label in enumerate(param_labels):
        param_values = array[..., i]
        min_array_i = param_values.min()
        max_array_i = param_values.max()
        logger.info(f"For parameter {param_label}, the minimun value is {min_array_i}, the maximum value is {max_array_i}")