"""Galactic-binary specialized stretch / RJ moves and supporting infrastructure."""

from __future__ import annotations

import json
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
from ..state import (
    GFState,
    ensure_cap_cell_fields,
    ensure_leaf_cap_fields,
    make_cap_edges,
)

__all__ = ["GBSpecialStretchMove"]

logger = logging.getLogger(__name__)

class _NoOpMempool:
    """CPU stand-in for ``cupy.get_default_memory_pool()`` — calls become no-ops."""

    def free_all_blocks(self):
        return


class _SharedProposalTables:
    """PARKED, NOT IN USE -- cold-chain proposal tables shared across proposals.

    Reached only through ``share_proposal_tables=True``, which currently raises
    ``NotImplementedError`` (see the TODO in
    :meth:`GBSpecialBase._ensure_proposal_tables`). The default path rebuilds
    both tables between every RJ step and its in-model sequence; this class is
    the plumbing for reusing one build across the moves of a cycle, kept so the
    option can be evaluated later rather than rewritten.

    A GB search iteration runs several GB moves back to back (fstat-birth ->
    replace -> prior-removal), and each of them wants the same two cold-chain
    products: the group-stretch friend table (frequency-sorted coordinates)
    and the info-matrix Cholesky table (frequency-sorted proposal factors).
    Rebuilding those per move is pure duplication -- the Cholesky side costs
    ~17 waveform evaluations per cold-chain source -- so they are built ONCE
    per larger iteration and reused.

    Lives on the run-shared :class:`AnalysisContainerArray` (see
    :func:`_shared_proposal_tables`), the same lazy-attach idiom as the
    run-shared ``DomainComputationGroupArray``, so no extra wiring is needed
    through the recipe and nothing is added to the picklable settings tree.

    ``iteration`` is the move's own ``self.time``: every move in the cycle
    proposes once per larger iteration, so they agree on it. A move that runs
    on a different cadence simply disagrees and triggers a rebuild -- extra
    work, never a stale table.

    The two products keep SEPARATE stamps so that, if this is ever enabled,
    the cheap friend sort and the expensive Cholesky build can run on
    different cadences.
    """

    def __init__(self):
        self.friends_iteration = None
        self.friends_coords_sorted = None
        self.infomat_iteration = None
        self.infomat_freqs_sorted = None
        self.infomat_chol_sorted = None

    def needs_friends(self, iteration) -> bool:
        return self.friends_iteration != iteration

    def needs_infomat(self, iteration, every) -> bool:
        if self.infomat_chol_sorted is None:
            return True
        if self.infomat_iteration == iteration:
            return False            # already built by another move this cycle
        return iteration % max(int(every), 1) == 0


def _shared_proposal_tables(acs, branch_name) -> _SharedProposalTables:
    """The run's ONE :class:`_SharedProposalTables` for ``branch_name``."""
    store = getattr(acs, "_gb_shared_proposal_tables", None)
    if store is None:
        store = {}
        acs._gb_shared_proposal_tables = store
    if branch_name not in store:
        store[branch_name] = _SharedProposalTables()
    return store[branch_name]


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
    make_routed_band_engine,
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

    def add(self, name: str, dt: float) -> None:
        """Checkpoint-style accumulation (span-free callers, e.g. the
        ``_mark`` boundaries inside ``_run_rj_step``)."""
        self.stages[name] = self.stages.get(name, 0.0) + float(dt)

    def report(self, total: float, top=None) -> str:
        # Top-level stages only: nested spans (buffer_build inside
        # run_proposal, ...) are reported but excluded from the
        # tracked/untracked accounting via the ``_TOP`` list.
        top = top or (
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


def _compact_index_ranges(indices, max_groups: int = 12) -> str:
    """``[0-3, 17, 40-44, ...]`` -- collapse an index list into runs.

    The leaf-cap log used to print every incremented band; on the cap-cell
    grid that is ``K`` times as many numbers (hundreds to thousands per
    line). Runs stay readable and still identify exactly what moved.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return "[]"
    idx = np.sort(idx)
    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [idx.size - 1]])
    groups = [
        f"{idx[s]}" if idx[s] == idx[e] else f"{idx[s]}-{idx[e]}"
        for s, e in zip(starts, ends)
    ]
    if len(groups) > max_groups:
        shown = ", ".join(groups[:max_groups])
        return f"[{shown}, ... +{len(groups) - max_groups} more runs]"
    return "[" + ", ".join(groups) + "]"


def _resolve_rj_flip_fraction(branch_name, kwarg_value, default=1.0):
    """Resolve ``rj_flip_fraction`` for a move (kwarg > env > ``default``).

    ``default`` is the stock/mode default the builder chose (the recipe
    passes 1.0 for search-cycle RJ moves and 0.1 for PE-cycle ones); a
    user env ``{BRANCH}_RJ_FLIP_FRACTION`` overrides it, an explicit kwarg
    overrides both. VGB is fixed-leaf (``nleaves_min == nleaves_max``, no
    RJ), so it gets NO RJ knob surface: the fraction is pinned to 1.0 and
    the env/default are never consulted (an explicit kwarg on a vgb move
    is rejected rather than silently ignored).
    """
    if str(branch_name).lower() == "vgb":
        if kwarg_value is not None:
            raise ValueError(
                "rj_flip_fraction is an RJ knob; the vgb branch is "
                "fixed-leaf (no RJ) and does not accept it."
            )
        return 1.0
    value = kwarg_value
    if value is None:
        value = os.environ.get(
            f"{str(branch_name).upper()}_RJ_FLIP_FRACTION", None
        )
    if value is None:
        value = default
    value = float(value)
    if not (0.0 < value <= 1.0):
        raise ValueError(
            f"rj_flip_fraction must be in (0, 1], got {value}."
        )
    return value


def _resolve_band_unit_stride(branch_name, ctor_value):
    """Resolve the band-unit stride ``band_units`` (env > ctor > default 2).

    A move built with an explicit ``band_units`` kwarg keeps it unless the
    env knob ``{BRANCH}_BAND_UNIT_STRIDE`` is set, which wins (the same
    env-wins convention as ``{BRANCH}_JUMP_FACTOR``). Stride ``k``
    partitions the bands into ``k`` units by ``band_index % k``; unit
    members are scheduled CONCURRENTLY, so same-unit bands always have
    ``k - 1`` closed bands between them. ``2`` (the default) is the
    legacy odd/even parity scheduling, bit-identical to the historical
    behavior. Values ``< 1`` are rejected.
    """
    env_val = os.environ.get(
        f"{str(branch_name).upper()}_BAND_UNIT_STRIDE", None
    )
    value = int(env_val) if env_val is not None else int(ctor_value)
    if value < 1:
        raise ValueError(f"band-unit stride must be >= 1, got {value}.")
    return value


def _tempering_open_remainder(start: int, units: int) -> int:
    """Band-index remainder class opened for tempering unit ``start``.

    The tempering grid selects interior bands ``arange(1, nb - 1)
    [start::units]``; those are exactly the bands with ``band % units ==
    (start + 1) % units`` (tempering begins at band 1). The residual
    open/close for the unit must expose the SAME class, so this mapping
    is the single source of truth. At ``units == 2`` it reproduces the
    legacy ``bool_remainder = 1 if start == 0 else 0``.
    """
    return (int(start) + 1) % int(units)


def _ortho_ll_summary(direct, credited, tol):
    """Per-unit bilinearity discrepancy summary for [GB_ORTHO_LL].

    PHYSICS PREMISE (user ruling, verified): FD inner product ~0 implies
    WDM inner product ~0, even within one wavelet layer, so same-unit
    cells scored concurrently in independent buffer components satisfy
    ``dll(h_i + h_j) = dll_i + dll_j - <h_i|h_j> ~ dll_i + dll_j``.
    ``direct`` is the realized per-cold-walker lnL delta on the overall
    parent residual across one unit (open -> proposals -> close);
    ``credited`` is the sum of the per-buffer (per-cell) lnL deltas the
    ledger accumulated for the same unit. Bilinearity/orthogonality
    failing (concurrent cells' windows interfering) makes them disagree.

    Returns ``dict(mean_abs=..., max_abs=..., worst_walker=...,
    flagged=bool)`` with ``flagged = max_abs > tol``.
    """
    direct = np.asarray(direct, dtype=float)
    credited = np.asarray(credited, dtype=float)
    disc = direct - credited
    abs_disc = np.abs(disc)
    k = int(abs_disc.argmax()) if abs_disc.size else 0
    max_abs = float(abs_disc[k]) if abs_disc.size else 0.0
    return {
        "mean_abs": float(abs_disc.mean()) if abs_disc.size else 0.0,
        "max_abs": max_abs,
        "worst_walker": k,
        "flagged": bool(max_abs > float(tol)),
    }


def _ortho_boundary_pairs(
    f0, walker_inds, band_inds, eligible, units, remainder, max_pairs=8
):
    """Closest-frequency same-unit cross-band source pairs (per walker).

    Boundary pairs are where the orthogonality premise (see
    :func:`lisatools.globalfit.moves.gbbands.check_band_support_separation`)
    is weakest: sources in DIFFERENT bands of the SAME concurrency unit
    (``band % units == remainder``) that are close in frequency --
    exactly the edge-source pairs whose FD supports
    (``get_N(f)/Tobs`` per side, the 2*get_N width rule's overhang
    analysis) can reach across the closed band(s) between them. Ranking
    by smallest ``|df|`` selects the worst such pairs. Within
    each walker's ``eligible`` sources (callers pass cold-chain alive),
    sort by ``f0`` and take every consecutive pair that crosses a band
    boundary; return the ``max_pairs`` smallest-``|df|`` pairs overall.

    All inputs are host numpy arrays. Returns ``(i_idx, j_idx)`` int
    arrays of row indices into the input arrays (empty when no
    cross-band pair exists).
    """
    f0 = np.asarray(f0, dtype=float)
    walker_inds = np.asarray(walker_inds)
    band_inds = np.asarray(band_inds)
    eligible = np.asarray(eligible, dtype=bool)
    sel = np.where(eligible & (band_inds % int(units) == int(remainder)))[0]
    pairs_i, pairs_j, pair_df = [], [], []
    for wv in np.unique(walker_inds[sel]):
        rows = sel[walker_inds[sel] == wv]
        if rows.size < 2:
            continue
        order = rows[np.argsort(f0[rows], kind="stable")]
        cross = np.where(np.diff(band_inds[order]) != 0)[0]
        pairs_i.append(order[cross])
        pairs_j.append(order[cross + 1])
        pair_df.append(f0[order[cross + 1]] - f0[order[cross]])
    if not pairs_i:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    i_all = np.concatenate(pairs_i)
    j_all = np.concatenate(pairs_j)
    df_all = np.concatenate(pair_df)
    keep = np.argsort(df_all, kind="stable")[: max(0, int(max_pairs))]
    return i_all[keep], j_all[keep]


def _resolve_inmodel_repeats(branch_name, class_name, kwarg_value, default):
    """Resolve a per-provenance-class in-model repeat count.

    Mirrors :func:`_resolve_rj_flip_fraction` (kwarg > env > ``default``):
    ``default`` is the mode default the caller chose (user ruling
    2026-08-15 — search: newborn 200 / survivor 25; PE: the move's
    ``num_repeat_proposals``, stock 100, for BOTH classes); a user env
    ``{BRANCH}_INMODEL_REPEATS_{CLASS}`` overrides it, an explicit kwarg
    (``inmodel_repeats_newborn`` / ``inmodel_repeats_survivor``)
    overrides both. Budgets are FIXED — never adaptive or early-exit —
    so the per-class chunk sequences keep the rigid shapes CUDA-graph
    capture needs.
    """
    value = kwarg_value
    if value is None:
        value = os.environ.get(
            f"{str(branch_name).upper()}_INMODEL_REPEATS_"
            f"{str(class_name).upper()}",
            None,
        )
    if value is None:
        value = default
    value = int(value)
    if value < 1:
        raise ValueError(
            f"inmodel_repeats_{class_name} must be >= 1, got {value}."
        )
    return value


def _inmodel_repeats_mode_defaults(branch_name, num_repeat_proposals):
    """``(newborn, survivor)`` mode defaults for the per-class budgets.

    USER RULING 2026-08-15: search mode polishes newborns hard and mature
    survivors lightly (200 / 25); PE uses the move's plain
    ``num_repeat_proposals`` (stock 100) for BOTH classes -- which also
    keeps lite presets (``num_repeat_proposals=2``) cheap. The mode is
    read from ``{BRANCH}_MODE`` (the same env that seeds the stock
    ``gb.mode`` field; default ``"pe"``); a builder that sets the mode
    programmatically should pass ``inmodel_repeats_*_default`` kwargs
    instead (see the ctor resolution).
    """
    if os.environ.get(f"{str(branch_name).upper()}_MODE", "pe") == "search":
        return 200, 25
    n = int(num_repeat_proposals)
    return n, n


def _split_by_newborn(merged, xp):
    """Partition a pooled-survivor dict by pick-time provenance.

    ``merged`` must carry a boolean ``"newborn"`` entry (True = the row
    was DEAD at pick time, i.e. an accepted birth; False = mature /
    death-rejected survivor). Returns ``[(class_name, class_dict), ...]``
    for the non-empty classes, newborns first, with the ``"newborn"`` key
    stripped from the class dicts and row order preserved within each
    class. Compression is one ``xp.where`` per class (a single device
    sync each on CuPy) followed by integer gathers.
    """
    nb_mask = merged["newborn"]
    rest = {k: v for k, v in merged.items() if k != "newborn"}
    out = []
    for cls_name, cls_mask in (("newborn", nb_mask), ("mature", ~nb_mask)):
        idx = xp.where(cls_mask)[0]
        if int(idx.size) == 0:
            continue
        out.append((cls_name, {k: v[idx] for k, v in rest.items()}))
    return out


def _buffer_fixed_capacity_active(sorter, kwargs) -> bool:
    """Whether ``_cached_get_buffer`` should use a fixed-capacity buffer.

    Fixed-capacity staging (user ruling 2026-08-14) applies to RJ
    proposal-phase buffers only: the sorter must be an RJ one
    (``sorter.rj_prop is not None`` — rj_fstat_search / rj_prior_removal /
    rj_replace / rj_*_pe) and the buffer must NOT carry the template twin
    (``use_template_arr`` — the tempering path, whose ~1200-cell chunks are
    far below the preload capacity; a capacity allocation there, doubled by
    the twin, would be a pure memory regression). Env gate
    ``GB_BUFFER_FIXED_CAPACITY`` (default "1"); "0" restores the
    drop+rebuild-on-size-change behavior verbatim.
    """
    return (
        os.environ.get("GB_BUFFER_FIXED_CAPACITY", "1") == "1"
        and getattr(sorter, "rj_prop", None) is not None
        and not kwargs.get("use_template_arr", False)
    )


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
        opt_snr_rej_samp_limit: GB SNR PRIOR BOUNDARY (user policy,
            default 5.0; applies in SEARCH AND PE -- it is a boundary in
            the high-dimensional prior): any proposed GB state -- RJ
            birth, replacement NEW side, or IN-MODEL update -- whose
            optimal SNR ``sqrt(h_h)`` falls below this limit is
            force-rejected (delta lnL -> -1e300) before the accept step;
            moving OUT of the violating region remains allowed
            (new-point test only). ONE limit shared with the optional
            detected test below.
        snr_rej_detected: Also test the DETECTED SNR ``d_h/sqrt(h_h)``
            against the same limit. Default OFF (user 2026-08-02: the
            observed-SNR gate is adjustable but not wanted by default --
            it fluctuates with the noise realization near threshold).
            ``None`` resolves from env ``GB_SNR_REJ_DETECTED``.
            Threaded through the BandSorter into the SubBandBuffer (single
            source of truth; the buffer's own default matches). NOT the
            same knob as ``snr_lim`` above.
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
        rj_removal_only: Search-mode pruning heuristic (2026-08-01). When
            ``True`` every BIRTH row of the RJ step is force-rejected
            (``curr_logp = -inf`` routes it through the existing ``keep``
            machinery, so it never reaches the likelihood kernel); DEATH
            rows run unchanged, with the death factors evaluated on THIS
            instance's ``rj_proposal_distribution`` logpdf as always. A
            death-only kernel is not self-reversible, so this is NOT a
            valid MH move on its own -- the USER has explicitly waived
            detailed-balance concerns for search mode. Pair it with a
            birth-capable RJ move in the same stage cycle; never use in PE.
        rj_replace: Search-mode REPLACEMENT proposal (2026-08-01). The
            dimension NEVER changes: each picked ALIVE leaf gets a fresh
            draw from ``rj_proposal_distribution`` and the move scores
            ``add(new) - add(old)`` against the old-source-exposed cell
            residual through :meth:`SubBandBuffer.get_replace_ll`
            (phase-maximized in search); on accept the standard swap
            (subtract old's template, add new's) is applied and ``inds``
            is untouched. Dead slots are never drawn for. Acceptance uses
            the phase-maximized ``add(old)`` as a comparison value only
            (a surviving source keeps its ORIGINAL parameters exactly),
            so this is a search heuristic, not exact MH (USER ruling).
            Mutually exclusive with ``rj_removal_only``.
        phase_maximize: If ``True``, marginalize over phase in the
            likelihood.
        gpus: GPU device list for this move (intra-node knob).
        band_units: Band-unit stride for the concurrent sub-band
            scheduling (env ``{BRANCH}_BAND_UNIT_STRIDE`` wins). Stride k
            partitions bands into k units by ``band_index % k``; a unit's
            bands are opened/scored CONCURRENTLY, so same-unit bands keep
            ``k - 1`` closed bands between them. Default 2 = the legacy
            odd/even parity, bit-identical. Honored by run_proposal AND
            run_tempering. PHYSICS RULING (user, verified premise): FD
            inner product ~0 implies WDM inner product ~0 even within one
            wavelet layer, so the concurrency constraint is ORTHOGONALITY
            (frequency separation), measured in FD-SUPPORT terms: the
            gap between same-unit bands vs the edge-source half-supports
            ``get_N(f_edge)/Tobs`` (``check_band_support_separation``).
            Hard enforcement lives at band-grid build time (the get_n
            builder, whose 2*get_N width rule guarantees stride 2); the
            ctor logs the verdict + minimum safe stride on WDM grids.
        num_band_preload: Staged-buffer slots PER RUN DEVICE
            (``GB_N_SUBBANDS``; total residency = this x n_gpus).
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
        opt_snr_rej_samp_limit=5.0,
        snr_rej_detected=None,
        rj_proposal_distribution=None,
        is_rj_prop=False,
        num_repeat_proposals=100,
        name=None,
        branch_name="gb",
        use_info_mat_proposal=True,
        swap_on_in_model=False,
        preserve_leaf_identity=None,
        use_prior_removal=False,
        rj_removal_only=False,
        rj_replace=False,
        phase_maximize=False,
        gpus=[],
        num_band_preload=20000,
        wdm_band_slab_layers=None,
        wdm_slab_guard_layers=1,
        run_swaps=True,
        temper_every_proposes=1,
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
        leaf_cap_ll_improve=True,
        leaf_cap_ndim=8.0,
        leaf_cap_min_iters=50,
        leaf_cap_ll_nsigma=3.0,
        leaf_cap_require_occupancy=True,
        leaf_cap_iter_only=False,
        leaf_cap_update=True,
        cap_divisor=None,
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
        # Cold-chain info-matrix Cholesky table: rebuilt between this move's RJ
        # step and its in-model sequence, then every source (any temp / walker)
        # borrows the nearest-in-frequency entry for the whole in-model
        # sequence.
        # Sources per batched information-matrix call when the table is built.
        self.infomat_table_batch = int(kwargs.get("infomat_table_batch", 2048))
        # NOT IMPLEMENTED -- reusing one table across proposals instead of
        # rebuilding per RJ proposal. See the TODO in _ensure_proposal_tables.
        self.share_proposal_tables = bool(kwargs.get("share_proposal_tables", False))
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
        # Search-mode RJ variants (see the class docstring): removal-only
        # prunes (births force-rejected in _run_rj_step); replace swaps an
        # alive leaf's parameters at fixed dimension (_run_replace_step).
        self.rj_removal_only = bool(rj_removal_only)
        self.rj_replace = bool(rj_replace)
        if self.rj_removal_only and self.rj_replace:
            raise ValueError(
                "rj_removal_only and rj_replace are mutually exclusive "
                "(build one move instance per mode)."
            )
        self.has_setup_group = False

        # GB-sampler verification instrumentation. When ``debug`` is on, the
        # ``_debug_*`` hooks below run band residual round-trip / get_ll
        # consistency checks and dump begin/middle/end band plots at the real
        # operation sites in ``run_proposal``. Off by default -> the hooks
        # early-return, so the production path is untouched.
        self.debug = bool(debug)
        # GB_JUMP_TRACE=1: per-propose census of the in-model f0 JUMP SIZE
        # against acceptance. Nothing else logs this. [GB_ACCEPT] gives the
        # acceptance rate but not the displacement behind it, and the two
        # only mean something together: 0.38 acceptance is produced both by
        # well-scaled steps that explore and by microscopic steps that are
        # accepted precisely because they change nothing. Accumulated
        # entirely on device (bincounts keyed on the temperature rung) and
        # pulled to host ONCE per propose, so it costs no per-repeat sync.
        self._jt = None
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
        # lnL-improvement cap gate -- THE DEFAULT (2026-08-12; takes
        # precedence over ``leaf_cap_iter_only``, so builders wiring the
        # fixed schedule must pass ``leaf_cap_ll_improve=False``): hold a
        # band's cap while its cold chain keeps finding a max ll better
        # than the stored best by >= leaf_cap_ndim/2 (D/2 = 4.0 for GBs).
        # ``False`` restores the legacy nsigma-spread + occupancy gate.
        self.leaf_cap_ll_improve = bool(leaf_cap_ll_improve)
        self.leaf_cap_ndim = float(leaf_cap_ndim)
        self._leaf_cap_enabled = leaf_cap_start is not None
        self.leaf_cap_min_iters = int(leaf_cap_min_iters)
        self.leaf_cap_ll_nsigma = float(leaf_cap_ll_nsigma)
        self.leaf_cap_require_occupancy = bool(leaf_cap_require_occupancy)
        # Iteration-only cap advancement (2026-08-01): when True the cap
        # increment gate in ``_update_band_leaf_caps`` is ONLY
        # ``iters >= leaf_cap_min_iters`` -- the lnL-plateau and occupancy
        # tests are skipped (the ``cap < nleaves_max`` guard stays). A
        # fixed-schedule annealing knob for search runs.
        self.leaf_cap_iter_only = bool(leaf_cap_iter_only)
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
        # Tempering cadence (user design 2026-08-14): a swap-enabled move
        # runs the band-swap stage only when at least this many TOTAL
        # branch proposes (every GBSpecial* propose() in the process,
        # shared census) have elapsed since the branch last tempered.
        # 1 = every eligible propose (legacy). Wired per-move by the
        # recipe: PE stacks carry several swap-enabled moves, so the
        # cadence turns tempering into a per-branch budget (e.g. every 3
        # gb proposes) instead of a per-move duty.
        self.temper_every_proposes = max(1, int(temper_every_proposes))

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
        self.opt_snr_rej_samp_limit = float(opt_snr_rej_samp_limit)
        if snr_rej_detected is None:
            snr_rej_detected = (
                os.environ.get("GB_SNR_REJ_DETECTED", "0") == "1")
        self.snr_rej_detected = bool(snr_rej_detected)
        logger.info(
            "%s: GB SNR prior boundary opt_snr_rej_samp_limit = %.2f "
            "(optimal SNR; search AND pe; births + replacement new-side "
            "+ in-model updates; detected-SNR test %s).",
            name, self.opt_snr_rej_samp_limit,
            "ON" if self.snr_rej_detected else "OFF",
        )

        # ------------------------------------------------------------------
        # LEAF-CAP CELL GRID (user design 2026-08-15)
        # ------------------------------------------------------------------
        # Sub-band widths are set by what the likelihood engine can run
        # CONCURRENTLY, which is far wider than the scale at which two GB
        # sources actually get confused (the posterior width). So the leaf
        # caps move off the band grid onto a finer "cap cell" grid: each
        # sub-band is split into ``cap_divisor`` equal pieces and the caps
        # are enforced per piece. NOTHING ELSE MOVES -- scheduling, units,
        # buffers, tempering, band shutoff and the cell-ll credit all stay
        # on the band grid; every cap cell is contained in exactly one band,
        # which is what keeps that separation cheap.
        #
        # ``cap_divisor == 1`` reproduces the pre-2026-08-15 per-band caps
        # BIT-IDENTICALLY (the cap grid IS the band grid and every helper
        # below short-circuits to the band arrays).
        # The knob lives on GBSettings (``gb.cap_divisor`` / GB_CAP_DIVISOR,
        # default 8) and is passed down by the recipe; the move's own
        # default is 1 so any other banded branch (VGB, tests, scripts)
        # keeps the per-band behaviour unless it opts in explicitly.
        self.cap_divisor = max(1, int(cap_divisor or 1))
        _be_host = _to_numpy(self.band_edges)
        self.cap_edges = self.xp.asarray(
            make_cap_edges(_be_host, self.cap_divisor)
        )
        self.num_cap_cells = self.num_bands * self.cap_divisor
        # per-band lower edge + cap-cell width, for the cell lookup
        self._cap_band_lo = self.xp.asarray(_be_host[:-1])
        self._cap_band_step = self.xp.asarray(
            (_be_host[1:] - _be_host[:-1]) / self.cap_divisor
        )
        #: live reference into ``band_info`` (per CAP CELL); see ``propose``
        self._cap_leaf_cap = None

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
        # Band-unit stride (``{BRANCH}_BAND_UNIT_STRIDE`` env wins over the
        # ctor kwarg): stride k partitions bands into k units by
        # ``band_index % k``; same-unit bands run CONCURRENTLY with k - 1
        # closed bands between them. 2 (default) = the legacy odd/even
        # parity scheduling, bit-identical. Honored by BOTH the proposal
        # unit loop (run_proposal) and the tempering unit loop
        # (run_tempering).
        self.band_units = _resolve_band_unit_stride(branch_name, band_units)
        # ORTHOGONALITY concurrency diagnostic (physics ruling; 2026-08-15
        # support-based form -- see check_band_support_separation): the
        # gap between same-unit bands is compared against the sum of the
        # edge-source FD half-supports get_N(f_edge)/Tobs. HARD
        # enforcement lives at band-grid BUILD time (the get_n builder
        # raises; the width rule guarantees stride 2 there); here the
        # ctor only LOGS the verdict + minimum safe stride -- the legacy
        # uniform grids fail this conservative envelope at high f by
        # design (their +-1-layer WDM windows, not the FD support, are
        # the operative bound; measured-safe) and must never be refused.
        # The [GB_ORTHO_LL] runtime monitor (default ON) is the
        # accuracy backstop for every grid. WDM basis only.
        if isinstance(self._basis_settings, WDMSettings):
            from .gbbands import check_band_support_separation

            try:
                _sep = check_band_support_separation(
                    band_edges,
                    float(self._basis_settings.Tobs),
                    self.band_units,
                    enforce=False,
                    context=f"{name or type(self).__name__} "
                            f"(branch {branch_name})",
                )
                logger.info(
                    "%s: support-based same-unit separation at stride %d: "
                    "passes=%s, min safe stride %s (sep_factor %.3g).",
                    name or type(self).__name__, self.band_units,
                    _sep["passes"], str(_sep["min_safe_stride"]),
                    float(_sep["sep_factor"]),
                )
            except Exception as exc:  # diagnostic only -- never block a move
                logger.warning(
                    "%s: support-separation diagnostic skipped: %r",
                    name or type(self).__name__, exc,
                )
        # Info-matrix jump scale (Gaussian draw through the Cholesky factor).
        self.jump_factor = float(jump_factor)
        # Env override (knob = capitalized field, branch-prefixed). The
        # measured in-model info-mat acceptance at the 0.005 default was
        # 0.95 -- steps ~0.5% of a posterior sigma; the optimal Gaussian
        # jump for a well-scaled Fisher is ~2.38/sqrt(d) with acceptance
        # ~0.23, so this knob is the direct mixing lever.
        _jf_env = os.environ.get(
            getattr(self, "branch_name", "gb").upper() + "_JUMP_FACTOR")
        if _jf_env:
            self.jump_factor = float(_jf_env)
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
        # Fraction of the RJ-eligible slots that receive a birth/death flip
        # attempt each proposal. The subset is drawn ONCE per proposal at
        # random WITHOUT replacement (see _apply_rj_flip_fraction); rows
        # outside it skip the flip while the in-model repeats still visit
        # them. 1.0 (default) = every slot, the historical behavior.
        # Kwarg ``rj_flip_fraction`` wins; env ``{BRANCH}_RJ_FLIP_FRACTION``
        # next; then the builder's ``rj_flip_fraction_default`` (the stock
        # recipe passes 1.0 for search-cycle RJ moves, 0.1 for PE-cycle).
        # VGB is fixed-leaf (nleaves_min == nleaves_max, NO RJ), so it gets
        # no RJ knob surface: pinned to 1.0, env/default ignored.
        self.rj_flip_fraction = _resolve_rj_flip_fraction(
            branch_name,
            kwargs.get("rj_flip_fraction", None),
            kwargs.get("rj_flip_fraction_default", 1.0),
        )
        # Per-class in-model repeat budgets (USER RULING 2026-08-15).
        # Survivors pooled for the end-of-unit in-model phase carry their
        # PICK-TIME provenance: "newborn" = the row was DEAD when picked
        # (an accepted birth); "mature" = it was ALIVE (a death-rejected
        # survivor; removal/replace pools are 100% mature by construction).
        # Search-mode defaults polish newborns hard (200) and survivors
        # lightly (25); PE uses the move's plain ``num_repeat_proposals``
        # budget (stock 100) for BOTH classes, which also keeps lite
        # presets (num_repeat_proposals=2) cheap. Resolution mirrors
        # ``rj_flip_fraction``: explicit kwarg > env
        # ``{BRANCH}_INMODEL_REPEATS_{NEWBORN,SURVIVOR}`` > the builder's
        # ``inmodel_repeats_{newborn,survivor}_default`` kwarg > the mode
        # default, with the mode read from ``{BRANCH}_MODE`` (the same env
        # that seeds the stock ``gb.mode`` field; default "pe"). Budgets
        # are FIXED, never adaptive/early-exit (fixed batch shapes are
        # what makes CUDA-graph capture possible later). Consumed by the
        # RJ round machinery only (see run_proposal / the direct-batch
        # in-model phase); pure in-model moves (``is_rj_prop=False``,
        # e.g. the search "in_model" move and VGB) keep
        # ``num_repeat_proposals`` untouched.
        _nb_mode_default, _surv_mode_default = _inmodel_repeats_mode_defaults(
            branch_name, self.num_repeat_proposals
        )
        self.inmodel_repeats_newborn = _resolve_inmodel_repeats(
            branch_name, "newborn",
            kwargs.get("inmodel_repeats_newborn", None),
            kwargs.get("inmodel_repeats_newborn_default", _nb_mode_default),
        )
        self.inmodel_repeats_survivor = _resolve_inmodel_repeats(
            branch_name, "survivor",
            kwargs.get("inmodel_repeats_survivor", None),
            kwargs.get(
                "inmodel_repeats_survivor_default", _surv_mode_default
            ),
        )
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
        # Shared info-matrix Cholesky table (cold-chain, frequency-sorted).
        # Persists ACROSS proposals -- unlike the friend table, rebuilding it
        # costs ~17 waveform evaluations per cold-chain source, so it is
        # refreshed on a cadence rather than every proposal.
        self._infomat_freqs_sorted = None
        self._infomat_chol_sorted = None
        # Set per proposal; guards the once-per-sorter table indexing.
        self._tables_indexed = False
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
        # fill by the walker's owning GPU split and runs each shard against
        # its own device-local comp replica.
        self._likelihood_engine = make_routed_band_engine(
            self._basis_settings,
            xp=self.xp,
            gb=self.gb,
            gb_fd_comp=self.gb_fd_comp,
            gb_wdm_comp=self.gb_wdm_comp,
            nchannels=acs.nchannels,
            tdi_channel_setup=self.waveform_kwargs.get("tdi_channel_setup"),
            df=self.df,
            start_freq_inds=self._parent_start_inds,
            data_length=acs.data_length,
        )
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

    @staticmethod
    def _debug_slab_kwargs(buffer_obj):
        """Slab-layout kwargs for DIRECT engine.fill_template calls in the
        GB_DEBUG hooks. The production fills go through
        ``SubBandBuffer._adjust_via_engine`` which forwards these; the debug
        hooks bypass it and were failing layout inference on narrow slab
        buffers ("templates flat size ... matches neither dense nor
        active")."""
        if getattr(buffer_obj, "band_slab_Nf", None) is not None:
            return dict(band_slab_Nf=buffer_obj.band_slab_Nf,
                        slab_min_f=buffer_obj.slab_min_f)
        return {}

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
                                  factor=+1, waveform_kwargs=self.waveform_kwargs,
                                  **self._debug_slab_kwargs(buffer_obj))
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
                                      factor=-1, waveform_kwargs=self.waveform_kwargs,
                                      **self._debug_slab_kwargs(buffer_obj))

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
                          factor=-1, waveform_kwargs=self.waveform_kwargs,
                          **self._debug_slab_kwargs(buffer_obj))
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
                              factor=+1, waveform_kwargs=self.waveform_kwargs,
                              **self._debug_slab_kwargs(buffer_obj))

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

            # The scratch is ONE slot; on the narrow-slab path pass the traced
            # slot's slab origin so the engine writes the slab layout the
            # reshape below expects.
            _slab_kw = {}
            if getattr(buffer_obj, "band_slab_Nf", None) is not None:
                _slab_kw = dict(
                    band_slab_Nf=Nf_a,
                    slab_min_f=buffer_obj.slab_min_f[[seq["slot"]]],
                )
            buffer_obj._likelihood_engine.fill_template(
                _Scratch(), params_phys,
                cp.zeros(n_src, dtype=cp.int32),
                band_sorter.band_N_vals[
                    cp.full(n_src, b, dtype=int)
                ],
                factor=+1, waveform_kwargs=self.waveform_kwargs,
                **_slab_kw,
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
                # Identity, not position: _run_rj_step re-subsets the batch
                # (flip-fraction gate) AFTER this select, so the positional
                # idx goes stale -- the accept hook looks the source up by
                # this id in ITS OWN row space.
                source_id=int(_to_numpy(picked["ids"])[idx]),
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
        except Exception as exc:
            # Never silent: a broken ranking here quietly pins every F-stat
            # reference to walker 0 for the whole run.
            logger.warning(
                "%s: could not rank cold walkers for the F-stat reference "
                "(%r); falling back to walker 0.", self.name, exc)
            return 0

    def run_proposal(self, model, state, band_sorter, band_temps):
        """One full pass of per-band proposals.

        Bands are partitioned into ``self.band_units`` units by
        ``band_index % band_units`` (stride-k generalization of the
        historical odd/even parity; ``GB_BAND_UNIT_STRIDE``, default 2 =
        bit-identical legacy). A unit's bands run CONCURRENTLY in
        independent buffer cells -- valid because same-unit bands keep
        ``band_units - 1`` closed bands of separation, sized against the
        edge-source FD supports (``check_band_support_separation``; the
        get_n 2*get_N width rule guarantees stride 2), so cross-cell
        template overlaps ``<h_i|h_j>`` are ~0 and likelihood deltas add
        by bilinearity (the orthogonality physics ruling; monitors:
        GB_ORTHO_CHECK / GB_ORTHO_LL_CHECK).

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

        # [GB_ORTHO_LL] bilinearity bookkeeping check (default OFF,
        # GB_ORTHO_LL_CHECK (default ON, user ruling 2026-08-15 -- the
        # cost is two extra parent-residual likelihood() evals per unit,
        # ~1.5 s/propose, negligible; =0 disables): per concurrent group
        # (one unit's simultaneously-scored cells), compare the sum of
        # per-buffer lnL deltas (the cold rows of ``ll_change_log``,
        # realized per-slab under the default GB_CELL_LL_CREDIT
        # crediting) against the realized delta on the OVERALL parent
        # residual across the unit's open -> proposals -> close.
        # Orthogonality is what makes the two agree (see
        # _ortho_ll_summary); this is the accuracy monitor for the
        # concurrent sub-band scheduling.
        _ortho_ll_on = os.environ.get("GB_ORTHO_LL_CHECK", "1") == "1"

        for unit_i in range(units):
            remainder = (start_unit + unit_i) % units

            if _ortho_ll_on:
                _oll_direct0 = _to_numpy(
                    model.analysis_container_arr.likelihood()
                ).copy()
                _oll_credit0 = _to_numpy(ll_change_log[0].sum(axis=-1)).copy()

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

            # RJ subsets include the dead (freshly-drawn) slots so births
            # can be proposed — EXCEPT removal-only, where _run_rj_step
            # force-rejects every birth: dead slots would only burn pick
            # rounds and scheduler cell counts on guaranteed -inf rows, so
            # the subset is alive-only, exactly like an in-model move (and
            # a cell with no alive sources never enters the scheduler; zero
            # alive anywhere -> get_subset returns None -> unit skipped).
            apply_inds = (not self.is_rj_prop) or self.rj_removal_only


            extra_bool = (
                (band_sorter.band_inds < self.num_bands - 1) & (band_sorter.band_inds > 0)
            ) if self.num_bands > 1 else None

            # AT-CAP RJ pick skip (2026-08-12, GB_RJ_SKIP_CAPPED=0 disables):
            # a birth into a band already holding cap[b] alive sources is
            # prior-forbidden, and _run_rj_step already keeps it away from
            # the likelihood kernel -- but the SCHEDULER still picked the
            # dead slots, burning pick rounds and diluting the acceptance
            # counters (a cap-saturated verify showed rj cold 0.0000 over
            # n=2104 all-impossible proposals). Exclude dead slots of
            # AT-CAP cells from the pick pool up front: deaths (alive
            # slots) stay proposable, so caps can still shed sources, and
            # a cell freed by a death mid-unit simply waits one unit for
            # its birth slots (the -inf enforcement in _run_rj_step remains
            # the correctness backstop either way). Counts are taken at
            # unit open, mirroring _run_rj_step's bincount arithmetic.
            self._rj_at_cap_mask = None
            if self.is_rj_prop and self._cap_leaf_cap is not None:
                xp_s = get_array_module(band_sorter.band_inds)
                # CAP-CELL occupancy (user design 2026-08-15): the census
                # and the caps live on the cap grid; scheduling below still
                # runs entirely on the band grid.
                _cap_inds = self._sorter_cap_cells(band_sorter)
                _flat, _cell_counts = self._cap_cell_counts(
                    band_sorter, _cap_inds
                )
                _cap = xp_s.asarray(self._cap_leaf_cap)
                _at_cap = self._cap_at_cap_mask(
                    band_sorter, _cell_counts, _cap, _flat, _cap_inds
                )
                # Per-source at-cap mask for the grouped in-model pool gate
                # (user rule 2026-08-13): at-cap cells never FREEZE sources
                # into the in-model pool — only below-cap cells add (birth)
                # then pool. Snapshot at unit open, same arithmetic as the
                # pick skip below.
                self._rj_at_cap_mask = _at_cap
                # LIVE-cap regime (user design 2026-08-14, default): at-cap
                # cells' dead rows STAY STAGED so a cell freed by an
                # accepted death can birth in the SAME unit — the live
                # per-round gate in _pick_sources throws them away while
                # the cell is at capacity, and the scheduler's finish
                # budget excludes them at unit open (countable-row init in
                # _run_band_unit) with cap-transition adjustments keeping
                # the budget exact (BandScheduler.add_counts). The
                # GB_RJ_LIVE_CAP_PICK=0 fallback restores the 2026-08-12
                # unit-open exclusion (one-unit wait for freed cells).
                _live_cap_on = (
                    not self.rj_removal_only
                    and not self.rj_replace
                    and os.environ.get("GB_RJ_LIVE_CAP_PICK", "1") == "1"
                )
                if (
                    not self.rj_removal_only
                    and not _live_cap_on
                    and os.environ.get("GB_RJ_SKIP_CAPPED", "1") == "1"
                ):
                    _rj_ok = band_sorter.inds | ~_at_cap
                    # Verification fingerprint (user request 2026-08-12):
                    # only DEAD slots of at-cap cells leave the pool (birth
                    # proposals); every alive slot in those cells stays
                    # proposable (deaths).
                    _dead_excluded = int((~_rj_ok).sum())
                    if _dead_excluded:
                        _capped_cells = int(xp_s.unique(
                            _flat[_at_cap & band_sorter.inds]).size)
                        _alive_at_cap = int((band_sorter.inds & _at_cap).sum())
                        logger.info(
                            f"{self.name}: rj at-cap skip -- {_dead_excluded} dead"
                            " (birth) slots excluded (their sub-band is"
                            " saturated across all"
                            f" {self.cap_divisor} cap cells); {_capped_cells}"
                            f" at-cap cap cells hold {_alive_at_cap} alive"
                            " slots that stay proposable (deaths)."
                        )
                    extra_bool = _rj_ok if extra_bool is None else (extra_bool & _rj_ok)
                elif _live_cap_on:
                    _n_reserve = int((~band_sorter.inds & _at_cap).sum())
                    if _n_reserve:
                        logger.info(
                            f"{self.name}: rj at-cap -- {_n_reserve} dead"
                            " (birth) slots staged as live-gated re-entry"
                            " reserve across"
                            f" {int(xp_s.unique(_flat[_at_cap & band_sorter.inds]).size)}"
                            " at-cap cap cells (births open the round a death"
                            " frees a cell in the sub-band)."
                        )

            # High-f barren-band shutoff enforcement (user design
            # 2026-08-14): dead rows of shut-off bands never enter the
            # subset — no draws consumed, no rounds, no counters. Alive
            # rows stay (deaths + in-model continue).
            _shut = getattr(self, "_rj_band_shutoff", None)
            if (
                _shut is not None and bool(_shut.any())
                and self._band_shutoff_enabled()
            ):
                xp_s2 = get_array_module(band_sorter.band_inds)
                _shut_dev = xp_s2.asarray(_shut)
                _shut_ok = (
                    band_sorter.inds
                    | ~_shut_dev[band_sorter.band_inds]
                )
                extra_bool = (
                    _shut_ok if extra_bool is None
                    else (extra_bool & _shut_ok)
                )

            # EARLY RJ FLIP (user design 2026-08-14): apply the flip
            # fraction where the pre-drawn proposal coordinates/logpdf
            # already live — BEFORE the scheduler is built — by excluding
            # the gated DEAD (birth) slots from the pick pool entirely,
            # exactly like the at-cap skip above. The old in-step gate
            # spent a full pick round + scheduler cell count on every row
            # it then discarded, which is why flip < 1 never saved wall
            # (identical pick census at flip 0.2 and 0.3 in production).
            # Excluded here, gated rows cost nothing: rounds shrink by
            # ~the flip fraction and every birth reaching the kernel is a
            # real proposal. ALIVE slots are NOT gated here — a picked
            # alive source must still pool for its in-model repeats even
            # when its death attempt is flip-gated (user rule 2026-08-12);
            # the death gate stays in _run_rj_step.
            if (
                self.is_rj_prop
                and not self.rj_replace
                and not self.rj_removal_only
                and self.rj_flip_fraction < 1.0
            ):
                xp_s = get_array_module(band_sorter.band_inds)
                dead_ids = xp_s.arange(band_sorter.num_sources)[
                    ~band_sorter.inds
                ]
                n_dead = int(len(dead_ids))
                if n_dead:
                    n_keep = max(
                        1, int(round(self.rj_flip_fraction * n_dead))
                    )
                    birth_ok = xp_s.zeros(
                        band_sorter.num_sources, dtype=bool
                    )
                    birth_ok[
                        dead_ids[xp_s.random.permutation(n_dead)[:n_keep]]
                    ] = True
                    _flip_ok = band_sorter.inds | birth_ok
                    extra_bool = (
                        _flip_ok if extra_bool is None
                        else (extra_bool & _flip_ok)
                    )

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
                # [GB_ORTHO] premise check (default OFF, GB_ORTHO_CHECK=1):
                # sample this unit's closest-frequency cross-band boundary
                # pairs and measure their normalized template overlaps
                # through the installed swap-ll kernels.
                self._run_ortho_premise_check(
                    model, band_sorter, units, remainder
                )

            # Close: re-subtract with (possibly updated) cold-chain coords.
            with _tspan(getattr(self, "_prop_timer", None), "unit_open_close"):
                self.add_cold_chain_sources_to_residual(
                    model, band_sorter, units=units, remainder=remainder
                )

            if _ortho_ll_on:
                _oll = _ortho_ll_summary(
                    _to_numpy(model.analysis_container_arr.likelihood())
                    - _oll_direct0,
                    _to_numpy(ll_change_log[0].sum(axis=-1)) - _oll_credit0,
                    float(os.environ.get("GB_ORTHO_LL_TOL", "0.05")),
                )
                logger.info(
                    "[GB_ORTHO_LL %s] unit %d (bands %% %d == %d): "
                    "|direct - credited| cold-walker lnL discrepancy mean "
                    "%.3e max %.3e (walker %d).",
                    self.name, unit_i, units, remainder,
                    _oll["mean_abs"], _oll["max_abs"], _oll["worst_walker"],
                )
                if _oll["flagged"]:
                    logger.warning(
                        "[GB_ORTHO_LL %s] unit %d: max per-walker "
                        "discrepancy %.3e exceeds GB_ORTHO_LL_TOL=%s -- the "
                        "sum of concurrent per-cell lnL deltas and the "
                        "realized parent-residual delta disagree beyond the "
                        "bilinearity/orthogonality allowance (concurrent "
                        "same-unit windows may be interfering).",
                        self.name, unit_i, _oll["max_abs"],
                        os.environ.get("GB_ORTHO_LL_TOL", "0.05"),
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
                    # Debug knob for the 2-GPU incremental-ll-drift hunt:
                    # deviceSynchronize() only syncs the CURRENT device, so
                    # kernels launched inside another shard's device context
                    # (cold-chain fills, per-shard scoring) may still be in
                    # flight when a later step reads their output via peer
                    # access. GB_MULTIGPU_SYNC_DEBUG=1 fences EVERY run
                    # device at each unit boundary: drift gone under the
                    # knob = cross-device stream race confirmed.
                    if os.environ.get("GB_MULTIGPU_SYNC_DEBUG", "0") == "1":
                        for _dev in (
                            getattr(model.analysis_container_arr, "gpus", None)
                            or []
                        ):
                            with self.xp.cuda.Device(int(_dev)):
                                self.xp.cuda.runtime.deviceSynchronize()
                    self.xp.cuda.runtime.deviceSynchronize()
                self.mempool.free_all_blocks()

        # Per-propose acceptance summary (rj = counters[0], in-model =
        # counters[1]; cold = temp index 0) -- the metric the in-model A/Bs
        # and flip-fraction tuning were blocked on.
        try:
            def _sum(a):
                return int(_to_numpy(a).sum())
            rj_p, rj_a = _sum(prop_counts[0]), _sum(acc_counts[0])
            im_p, im_a = _sum(prop_counts[1]), _sum(acc_counts[1])
            rj_pc, rj_ac = _sum(prop_counts[0][0]), _sum(acc_counts[0][0])
            im_pc, im_ac = _sum(prop_counts[1][0]), _sum(acc_counts[1][0])
            logger.info(
                "[GB_ACCEPT %s] rj cold %d/%d (%.4f) all %d/%d (%.4f) | "
                "in-model cold %d/%d (%.4f) all %d/%d (%.4f)",
                self.name, rj_ac, rj_pc, rj_ac / max(rj_pc, 1),
                rj_a, rj_p, rj_a / max(rj_p, 1),
                im_ac, im_pc, im_ac / max(im_pc, 1),
                im_a, im_p, im_a / max(im_p, 1))
            kc = getattr(self, "_im_kind_counts", None)
            if kc:
                parts = "; ".join(
                    f"{k}: cold {r[3]}/{r[2]} ({r[3] / max(r[2], 1):.4f}) "
                    f"all {r[1]}/{r[0]} ({r[1] / max(r[0], 1):.4f})"
                    for k, r in sorted(kc.items()))
                logger.info("[GB_ACCEPT %s] in-model by proposal type -- %s "
                            "(jump_factor=%.4g)", self.name, parts,
                            self.jump_factor)
                self._im_kind_counts = {}
            # GB_JUMP_TRACE: emitted next to [GB_ACCEPT] so the rate and the
            # displacement that produced it are read together.
            self._jump_trace_report()
            self._infomat_warned = False        # re-arm the route indicator
            # Denominator split (user request 2026-08-14): the headline is
            # MH acceptance among VIABLE births -- rows the sampler
            # actually compared -- with the auto-reject classes broken
            # out. With the live cap gate on, "capped" should read ~0.
            sp = getattr(self, "_rj_split", None)
            if sp:
                v, vc = sp.get("viable", 0), sp.get("viable_cold", 0)
                ba, bac = sp.get("birth_acc", 0), sp.get("birth_acc_cold", 0)
                logger.info(
                    "[GB_ACCEPT rj-split %s] births %d: viable %d "
                    "(acc %d = %.4f; cold %d/%d = %.4f) | gated: prior %d "
                    "oob %d capped %d | scored-dropped: snr %d kernel %d | "
                    "deaths %d (acc %d)",
                    self.name, sp.get("births", 0), v,
                    ba, ba / max(v, 1), bac, vc, bac / max(vc, 1),
                    sp.get("prior", 0), sp.get("oob", 0),
                    sp.get("capped", 0), sp.get("snr", 0),
                    sp.get("kernel", 0), sp.get("deaths", 0),
                    sp.get("death_acc", 0))
                self._rj_split = None
            # High-f band shutoff bookkeeping — OCCUPANCY-based (user key
            # change 2026-08-15): cold-chain per-band occupancy, max over
            # walkers, once per iteration on the designated move (log
            # contract line emitted inside _update_band_shutoff).
            if self._band_shutoff_enabled():
                self._update_band_shutoff(
                    self._band_occupancy_cold_max(new_state))
        except Exception:  # diagnostics must never kill a propose
            pass
        # Per-propose F-stat peak census: how many birth draws each peak
        # received (settings lever: if peaks are drawn many times with no
        # acceptance, futility-based retirement / flip fraction can shrink
        # rj_step without losing coverage).
        try:
            census = self._stacked_for_census()
            if census is not None:
                c = census.pop_draw_counts()
                tot = int(c.sum())
                if tot:
                    logger.info(
                        "[FSTAT_PEAKS %s] %d draws over %d/%d peaks; "
                        "per-peak mean %.1f median %d max %d (peak #%d); "
                        "never-drawn %d",
                        self.name, tot, int((c > 0).sum()), len(c),
                        tot / max(len(c), 1), int(np.median(c)),
                        int(c.max()), int(c.argmax()), int((c == 0).sum()))
        except Exception:
            pass

        return ll_change_log, prop_counts, acc_counts

    def _stacked_for_census(self):
        """Find the StackedFStatProposal4D inside the rj birth container.

        Cached per install (``_install`` resets it); returns None when the
        births come from the prior (no census surface).
        """
        obj = getattr(self, "_stacked_census_obj", "unset")
        if obj != "unset":
            return obj
        found = None
        seen = set()

        def walk(o, depth=0):
            nonlocal found
            if o is None or depth > 5 or id(o) in seen or found is not None:
                return
            seen.add(id(o))
            if hasattr(o, "pop_draw_counts"):
                found = o
                return
            for a in ("priors_in", "priors", "components", "base", "dist",
                      "distribution"):
                v = getattr(o, a, None)
                if isinstance(v, dict):
                    for vv in v.values():
                        walk(vv, depth + 1)
                elif isinstance(v, (list, tuple)):
                    for vv in v:
                        walk(vv[-1] if isinstance(vv, tuple) else vv,
                             depth + 1)
                elif v is not None:
                    walk(v, depth + 1)

        cont = self.rj_proposal_distribution
        if isinstance(cont, dict):
            cont = cont.get(self.branch_name)
        walk(cont)
        self._stacked_census_obj = found
        return found

    def _debug_sync_all_devices(self, model):
        """GB_MULTIGPU_SYNC_DEBUG=1: fence EVERY run device.

        The unit-boundary/fill fences (below and in gbbands) cover the
        WRITE side; this covers the READ side — called immediately before
        each tracked-vs-true ll comparison in ``propose``. Discriminator
        for the shard-1 drift: the parent residual is WALKER-sharded while
        the buffer is BAND-sharded, so closes issue cross-device writes
        (commonly device-0 context -> shard-1 walkers' residuals). If
        ``likelihood()`` reads those rows before the writing device's
        stream flushes, exactly the shard-1 walkers drift and the state
        settles by the next start check — the observed signature. Drift
        gone under this knob = read-side race confirmed (and the knob is
        the interim mitigation); unchanged = the credit/accounting side.
        """
        if os.environ.get("GB_MULTIGPU_SYNC_DEBUG", "0") != "1":
            return
        if not self.backend.uses_cupy:
            return
        for _dev in (
            getattr(model.analysis_container_arr, "gpus", None) or []
        ):
            with self.xp.cuda.Device(int(_dev)):
                self.xp.cuda.runtime.deviceSynchronize()
        self.xp.cuda.runtime.deviceSynchronize()

    # ============================================================
    # ONE SubBandBuffer cache PER COMP GROUP, shared by EVERY GB move
    # (user ruling 2026-08-14). WHY SHARED: rj_fstat_search,
    # rj_prior_removal, rj_fstat_pe/rj_prior_pe and the in-model moves all
    # build INTERCHANGEABLE buffers -- same computation objects, same slab
    # geometry; the only real construction differences (template twin for
    # tempering, rj-vs-inmodel edge windows) ride in the cache ENTRY key.
    # Per-move caches meant every move held its own multi-GB allocation
    # simultaneously: in job 183 rj_fstat_search's 32k-slot buffer AND
    # rj_prior_removal's 11.5k-slot buffer were both resident (~45 GB of
    # duplicated slabs) when the fill OOM'd. Moves propose SEQUENTIALLY, so
    # sharing is race-free; when consecutive moves need different sizes the
    # single-buffer policy drops+rebuilds (~1-2 s per alternation) instead
    # of holding both. Scope key = id(comp group): VGB moves carry their
    # own comps and therefore their own scope automatically -- GB and VGB
    # never share a buffer.
    # ============================================================
    _shared_buffer_caches: dict = {}
    # Shared per-branch propose census + last-temper marker for the
    # tempering cadence (user design 2026-08-14). Class-level ON PURPOSE:
    # "every N proposes" counts ALL GBSpecial* propose() calls of the
    # branch across every move instance in the process.
    _branch_propose_counts: dict = {}
    _branch_last_temper: dict = {}

    @property
    def _buffer_cache_scope(self):
        comp = (self.gb_wdm_comp if getattr(self, "gb_wdm_comp", None)
                is not None else getattr(self, "gb_fd_comp", None))
        return id(comp)

    @property
    def _prop_buffer_cache(self):
        return GBSpecialBase._shared_buffer_caches.get(
            self._buffer_cache_scope)

    @_prop_buffer_cache.setter
    def _prop_buffer_cache(self, value):
        if value is None:
            GBSpecialBase._shared_buffer_caches.pop(
                self._buffer_cache_scope, None)
        else:
            GBSpecialBase._shared_buffer_caches[
                self._buffer_cache_scope] = value

    @property
    def num_band_preload_total(self) -> int:
        """Total staged-buffer slots: ``GB_N_SUBBANDS`` is PER RUN DEVICE
        (user ruling 2026-08-14). The buffer is band-sharded across the
        run devices, so the knob states each GPU's residency/memory
        budget and total residency scales with the allocation."""
        gpus = (getattr(self.gb, "gpus", None)
                if self.backend.uses_cupy else None)
        return int(self.num_band_preload) * (len(gpus) if gpus else 1)

    def _cached_get_buffer(self, sorter, acs, specials, **kwargs):
        """SubBandBuffer reuse: ONE live buffer per construction signature.

        A call whose slot count matches the cached buffer performs a FULL
        rebind (the existing ``inds_fill``/``buffer_obj`` path —
        construction minus allocation). On a size mismatch there are two
        regimes:

        * **Fixed-capacity RJ buffers** (``GB_BUFFER_FIXED_CAPACITY=1``,
          default; user ruling 2026-08-14): proposal-phase buffers of RJ
          sorters (``sorter.rj_prop is not None``, no template twin) are
          built ONCE with ``alloc_capacity = num_band_preload_total`` slots
          and any later unit with ``k <= capacity`` cells is RESIZE-REBOUND
          into the front of that allocation
          (``SubBandBuffer.resize_to`` + full rebind) — never dropped. The
          RJ unit sizes are alive-cells-only and drift every iteration
          (13,043 / 11,363 / ... never repeating), so the old
          size-keyed drop+rebuild rebuilt a ~12 GB buffer on EVERY unit
          (~120 s/propose measured). Only a ``k > capacity`` request (never
          produced by the scheduler, which caps ``n_slots`` at the preload)
          drops and rebuilds.
        * **Everything else** (non-RJ sorters; RJ tempering buffers with the
          template twin, whose ~1200-cell chunks are far below the preload
          capacity): today's behavior verbatim — a size change drops the
          cached buffer and rebuilds at the new size, so at most one buffer
          per kwargs signature is ever resident.

        The template twin rides in the signature: proposal-phase buffers
        never carry it (its guarded fill paths are tempering-only). Under
        ``GB_BUFFER_PERSIST=1`` the cache survives across proposals (see
        ``_buffer_cache_teardown``); ``GB_BUFFER_CACHE_PER_SIZE=1`` restores
        per-size caching; ``GB_BUFFER_FIXED_CAPACITY=0`` restores the
        drop+rebuild-on-any-size-change behavior for RJ buffers too.
        """
        cache = getattr(self, "_prop_buffer_cache", None)
        if cache is None:
            cache = self._prop_buffer_cache = {}
        k = int(specials.shape[0])
        fixed_cap = _buffer_fixed_capacity_active(sorter, kwargs)
        build_kwargs = dict(kwargs)
        if fixed_cap:
            # Capacity = the staged-slot budget, clamped by the move's STATIC
            # maximum cell count (every (temp, walker, band) cell) so small
            # runs (lite gates, CPU tests) never allocate a preload-sized
            # buffer for a handful of cells. Every scheduler binding has
            # k <= min(preload_total, n_cells) <= this capacity, so a
            # resize-rebind always fits and the clamp never reintroduces the
            # drop+rebuild path. Both terms are run constants -> the cache
            # signature is stable across units.
            _max_cells = (
                int(self.ntemps) * int(self.nwalkers)
                * (len(self.band_edges) - 1)
            )
            build_kwargs["alloc_capacity"] = min(
                int(self.num_band_preload_total), _max_cells
            )
        # rj-vs-inmodel buffers differ in edge windows (N/4 widening on RJ
        # sorters), so the sorter's rj flag is part of the entry key even
        # though the cache itself is shared across moves.
        sig = ((bool(getattr(sorter, "rj_prop", False)),)
               + tuple(sorted(build_kwargs.items())))
        # ONE live buffer per construction signature (user design
        # 2026-08-13): the allocation follows the current unit's slot count,
        # so a size change drops the old buffer (the pool reuses its blocks)
        # and rebuilds -- total buffer memory is bounded by one buffer per
        # signature. Keying on size instead let one buffer per DISTINCT unit
        # size accumulate for the life of the run (three 10000-slot buffers
        # were resident when the 40 GB device OOM'd, 2026-08-13 3-mo run).
        # GB_BUFFER_CACHE_PER_SIZE=1 restores per-size keys (no rebuild on
        # size change) LRU-capped at GB_BUFFER_CACHE_MAX, if the rebuild
        # churn of alternating unit sizes ever costs more than it saves.
        # Never combined with fixed capacity: per-size keys would hold one
        # CAPACITY-sized allocation per distinct unit size — the exact
        # accumulation the one-live-buffer policy exists to prevent (and
        # resize-rebind already removes the churn per_size was for).
        per_size = (
            os.environ.get("GB_BUFFER_CACHE_PER_SIZE", "0") == "1"
            and not fixed_cap
        )
        key = (k, sig) if per_size else sig
        buf = cache.get(key)
        if buf is not None and int(buf.num_bands_now) != k:
            _cap = getattr(buf, "alloc_capacity", None)
            if fixed_cap and _cap is not None and int(_cap) >= k:
                # Fixed-capacity hit at a different size: keep the buffer —
                # the rebind below resize-rebinds it (allow_resize=True).
                pass
            else:
                # Capacity insufficient (or capacity mode off): drop and
                # rebuild at the new size — today's behavior.
                del cache[key]
                buf = None
        if buf is None:
            buf = sorter.get_buffer(
                acs, specials,
                timer=getattr(self, "_prop_timer", None),
                **build_kwargs,
            )
            cache[key] = buf
            self._prop_buffer_builds = getattr(self, "_prop_buffer_builds", 0) + 1
            if per_size:
                max_sigs = int(os.environ.get("GB_BUFFER_CACHE_MAX", "8"))
                while len(cache) > max_sigs:
                    cache.pop(next(iter(cache)))
            if self.backend.uses_cupy:
                _cap = getattr(buf, "alloc_capacity", None)
                logger.info(
                    "%s: buffer build (%s%s); GPU pool used "
                    "%.2f / total %.2f GB; %s", self.name,
                    (f"{int(_cap)}-slot alloc, {k} bound" if _cap is not None
                     else f"{k} slots"),
                    ", " + repr(sig) if sig else "",
                    self.mempool.used_bytes() / 1e9,
                    self.mempool.total_bytes() / 1e9,
                    self._device_mem_summary())
        else:
            # Re-insert on hit so dict order stays LRU (per-size mode).
            cache.pop(key)
            cache[key] = buf
            sorter.get_buffer(
                acs, specials,
                inds_fill=self.xp.arange(k),
                buffer_obj=buf,
                allow_resize=fixed_cap,
                timer=getattr(self, "_prop_timer", None),
            )
        # Speed-diagnosis plumbing (user directive 2026-08-15): the buffer
        # carries the CURRENT propose's timer so its hot methods (get_ll /
        # fills) and the shard router can attribute their internals
        # (gll_* / route_* / fill_* spans). Refreshed every bind — a
        # persistent cached buffer must never hold a previous propose's
        # timer.
        buf._prop_timer = getattr(self, "_prop_timer", None)
        return buf

    def _device_mem_summary(self) -> str:
        """Per-device ``used/total GB`` from ``memGetInfo`` (all devices).

        Device-wide truth, unlike the cupy pool stats: includes the other
        devices' pools, raw C++/CUDA allocations (sig-het stashes, kernel
        scratch), and other processes on the card.
        """
        if not self.backend.uses_cupy:
            return "cpu"
        rt = self.xp.cuda.runtime
        parts = []
        main_dev = rt.getDevice()
        try:
            for d in range(rt.getDeviceCount()):
                with self.xp.cuda.Device(d):
                    free, tot = rt.memGetInfo()
                parts.append(f"dev{d} {(tot - free) / 1e9:.1f}/{tot / 1e9:.1f}")
        finally:
            rt.setDevice(main_dev)
        return "device used/total GB: " + ", ".join(parts)

    def _buffer_cache_teardown(self):
        """Proposal-exit buffer bookkeeping: sweep pools, maybe drop the cache.

        Under ``GB_BUFFER_PERSIST=1`` (default) the cached buffer ACAs
        SURVIVE the proposal: the next proposal's same-signature
        ``_cached_get_buffer`` call takes the full-rebind path (refill in
        place -- no reconstruction of thousands of per-cell containers),
        which is where the ~16 s/propose ``buffer_build``/``temper_buffer``
        cost went. The device pools are still synced and swept (only
        pool-cached FREE blocks are released; live buffer allocations are
        held by the cached arrays).

        ``GB_BUFFER_PERSIST=0`` restores the strict proposal-scoped
        contract (memory-lifecycle rule): buffers dropped and pools swept
        at every proposal exit so the memory is available to the other
        modules' moves between GB proposals. Engine bindings and shard
        views cache on the buffers themselves and follow the cache's
        lifetime in both modes.
        """
        persist = os.environ.get("GB_BUFFER_PERSIST", "1") == "1"
        cache = getattr(self, "_prop_buffer_cache", None)
        n_builds = getattr(self, "_prop_buffer_builds", 0)
        devices = set()
        for buf in (cache or {}).values():
            for dev in (getattr(buf, "gpus", None) or []):
                devices.add(int(dev))
        if not persist:
            self._prop_buffer_cache = None
        self._prop_buffer_builds = 0
        if not self.backend.uses_cupy:
            return
        used = self.mempool.used_bytes() / 1e9
        total = self.mempool.total_bytes() / 1e9
        logger.info(
            "%s: buffer lifecycle -- %d allocation(s) this propose "
            "(%d cached signature(s)); GPU pool used %.2f / total %.2f GB; "
            "%s.", self.name, n_builds, len(cache or {}), used, total,
            self._device_mem_summary(),
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

    def _cell_ll_state_init(self, scheduler, band_temps=None):
        """Per-unit state for buffer before/after cell ll crediting."""
        xp = scheduler.xp
        n = scheduler.n_slots
        spec = scheduler.slot_specials
        return {
            "ll0": xp.zeros(n),
            "led0": xp.zeros(n),
            "rep0": xp.zeros(n, dtype=int),
            "spec": xp.zeros(n, dtype=spec.dtype),
            "open": xp.zeros(n, dtype=bool),
            "band_temps": band_temps,
            "n_done": 0,
            "sum_abs_mm": 0.0,
            "max_mm": 0.0,
            "max_excess": 0.0,
            "worst": None,
        }

    def _cell_ll_open(self, st, buffer_obj, slots, specials,
                      ll_change_log, prop_counts):
        """Snapshot slab ll + ledger/repeat baselines for freshly filled slots."""
        if len(slots) == 0:
            return
        lls = buffer_obj.band_likelihoods(source_only=True)
        t_i, w_i, b_i = unpack_special_index(specials, self.nwalkers)
        st["ll0"][slots] = lls[slots]
        st["led0"][slots] = ll_change_log[t_i, w_i, b_i]
        st["rep0"][slots] = prop_counts[1][t_i, w_i, b_i]
        st["spec"][slots] = specials
        st["open"][slots] = True

    # ---- GB_JUMP_TRACE ---------------------------------------------------
    # Jump size vs acceptance, the pairing no existing log line reports.
    # [GB_ACCEPT] gives the rate; the rate alone is ambiguous, because 0.38
    # is produced both by well-scaled steps that explore and by microscopic
    # steps accepted precisely because they change nothing. |df0| is carried
    # in FOURIER BINS, the unit in which the Fisher width (0.551/SNR) and the
    # sig-het trust gate (dphase/2pi = 0.0796 bins) are both naturally
    # expressed, so the report puts all three on one scale.
    _JT_NSNR, _JT_SNR_W = 12, 10.0        # SNR bins of width 10, 0..120

    #: Per-source ``information_matrix`` cost above which the call is
    #: assumed to have fallen through to the chunked delegate. Sits between
    #: the two measured rates (~2.4 ms sig-het, ~46 ms chunked) with an
    #: order of magnitude of headroom either side, so it does not fire on
    #: ordinary slow blocks.
    _INFOMAT_SLOW_MS = 15.0

    def _infomat_route_check(self, dt, nrows, *, fast_wired, comp):
        """Warn ONCE per propose if the info matrix missed the sig-het route.

        ``fast_wired`` records whether the caller supplied the slot-space
        routing the fast leg needs; a sig-het comp with the reference NOT
        live still answers, just via its chunked delegate, which is exactly
        the silent case this exists to surface.
        """
        if nrows <= 0 or getattr(self, "_infomat_warned", False):
            return
        ms = 1e3 * dt / nrows
        if ms < self._INFOMAT_SLOW_MS:
            return
        self._infomat_warned = True
        logger.warning(
            "[GB_INFOMAT %s] information_matrix cost %.1f ms/source over %d "
            "rows -- the sig-het route is ~2.4 ms/source and the chunked "
            "delegate ~46, so this looks like a FALL-THROUGH (sig-het comp: "
            "%s; slot routing wired: %s). Check that "
            "setup_in_model_likelihood ran BEFORE the proposal Cholesky; "
            "built the other way round the reference is not live and the "
            "call silently takes the chunked path.",
            self.name, ms, nrows, hasattr(comp, "chunked"), fast_wired)

    def _jump_trace_new(self, nt):
        xp = self.xp
        z = lambda k: xp.zeros(k, dtype=xp.float64)
        return dict(nt=nt, n=z(nt), n_acc=z(nt), s_d=z(nt), s_d_acc=z(nt),
                    s_d2=z(nt), snr_n=z(self._JT_NSNR),
                    snr_acc=z(self._JT_NSNR), snr_d=z(self._JT_NSNR))

    def _jump_trace_accum(self, new, cur, accept, t_idx, keep_idx, h_h):
        """One repeat's (|df0|, accepted) into the census; device-only.

        Five bincounts keyed on the temperature rung, three on an SNR bin.
        No host sync here -- the pull happens once per propose in
        :meth:`_jump_trace_report`.
        """
        if os.environ.get("GB_JUMP_TRACE", "0") != "1":
            return
        fc = self._f0_col
        if fc is None or new.shape[0] == 0:
            return          # cupy.bincount raises on zero-size input
        xp = self.xp
        ti = xp.asarray(t_idx).astype(xp.int32)
        nt = int(getattr(self, "ntemps", 0)) or (int(ti.max()) + 1)
        if self._jt is None or self._jt["nt"] != nt:
            self._jt = self._jump_trace_new(nt)
        S = self._jt
        d = xp.abs(new[:, fc] - cur[:, fc]) * 1e-3 / self.df       # bins
        ac = accept.astype(d.dtype)
        S["n"] += xp.bincount(ti, minlength=nt).astype(d.dtype)
        S["n_acc"] += xp.bincount(ti, weights=ac, minlength=nt)
        S["s_d"] += xp.bincount(ti, weights=d, minlength=nt)
        S["s_d_acc"] += xp.bincount(ti, weights=d * ac, minlength=nt)
        S["s_d2"] += xp.bincount(ti, weights=d * d, minlength=nt)
        # Per-source SNR split: the loud end is exactly where production
        # freezes (within-walker scatter 0.0000 bins at SNR > 40), so an
        # aggregate rate would hide the thing this probe exists to see.
        if h_h is None or keep_idx is None or int(keep_idx.shape[0]) == 0:
            return
        snr = xp.sqrt(xp.clip(xp.asarray(h_h).real, 0.0, None)).ravel()
        dk, ak = d[keep_idx], ac[keep_idx]
        k = min(int(snr.shape[0]), int(dk.shape[0]))
        if k == 0:
            return
        b = xp.clip((snr[:k] / self._JT_SNR_W).astype(xp.int32),
                    0, self._JT_NSNR - 1)
        S["snr_n"] += xp.bincount(b, minlength=self._JT_NSNR).astype(d.dtype)
        S["snr_acc"] += xp.bincount(b, weights=ak[:k], minlength=self._JT_NSNR)
        S["snr_d"] += xp.bincount(b, weights=dk[:k], minlength=self._JT_NSNR)

    def _jump_trace_report(self):
        """Pull the census to host, log it, and clear. Once per propose."""
        S = self._jt
        if S is None:
            return
        self._jt = None
        n = _to_numpy(S["n"])
        if n.sum() <= 0:
            return
        na, sd = _to_numpy(S["n_acc"]), _to_numpy(S["s_d"])
        sda, sd2 = _to_numpy(S["s_d_acc"]), _to_numpy(S["s_d2"])
        den = np.maximum(n, 1.0)
        acc, mean_d = na / den, sd / den
        rms_d = np.sqrt(sd2 / den)
        mean_da = sda / np.maximum(na, 1.0)
        gate = float(self.sighet_trust_dphase) / (2.0 * np.pi)
        logger.info(
            "[GB_JUMP %s] in-model |df0| in FOURIER BINS (trust gate %.4f, "
            "jump_factor %.3g) -- rung: n acc mean rms mean_accepted | %s",
            self.name, gate, self.jump_factor,
            "; ".join(f"T{i}: {int(n[i])} {acc[i]:.3f} {mean_d[i]:.4f} "
                      f"{rms_d[i]:.4f} {mean_da[i]:.4f}"
                      for i in np.nonzero(n > 0)[0]))
        sn, sa = _to_numpy(S["snr_n"]), _to_numpy(S["snr_acc"])
        sdd = _to_numpy(S["snr_d"])
        if (sn > 0).any():
            logger.info(
                "[GB_JUMP %s] by source SNR -- %s", self.name,
                "; ".join(
                    f"{int(i * self._JT_SNR_W)}-"
                    f"{int((i + 1) * self._JT_SNR_W)}: n {int(sn[i])} "
                    f"acc {sa[i] / max(sn[i], 1.0):.3f} "
                    f"mean|df0| {sdd[i] / max(sn[i], 1.0):.4f} "
                    f"(Fisher {0.551 / ((i + 0.5) * self._JT_SNR_W):.4f})"
                    for i in np.nonzero(sn > 0)[0]))

    def _cell_ll_finalize(self, st, buffer_obj, slots, ll_change_log,
                          prop_counts):
        """Credit switched-out cells with the realized slab ll difference.

        The smallest running unit (2026-08-12 user design): a cell's stay in
        the buffer is fill -> ll -> rj -> in-model repeats -> ll. The
        before/after slab difference replaces the cell's ACCUMULATED sampled
        lls in ``ll_change_log``, so the ledger tracks the residual the
        buffer actually holds; the sampled-vs-realized difference is checked
        per in-model repeat right here (and only here).
        """
        if len(slots) == 0:
            return
        xp = get_array_module(st["ll0"])
        lls = buffer_obj.band_likelihoods(source_only=True)
        spec = st["spec"][slots]
        t_i, w_i, b_i = unpack_special_index(spec, self.nwalkers)
        actual = lls[slots] - st["ll0"][slots]
        sampled = ll_change_log[t_i, w_i, b_i] - st["led0"][slots]
        nrep = prop_counts[1][t_i, w_i, b_i] - st["rep0"][slots]
        mm = actual - sampled
        ll_change_log[t_i, w_i, b_i] = st["led0"][slots] + actual
        rate = xp.abs(mm) / xp.maximum(nrep, 1)
        # Temperature-scaled allowance (the tiered-accuracy ruling): hot
        # rungs sample displacements where the sig-het error is ALLOWED to
        # grow -- the trust region is the hard gate out there -- so holding
        # every rung to the cold-chain floor buries real cold offenders in
        # hot-rung noise (a 124.5 at temp 23 is expected; the same number
        # at temp 0 is a bug). allowed ~ tol/beta; beta=1 keeps the floor.
        # GB_CELL_LL_TEMP_SCALED=0 restores the uniform floor.
        tol = float(os.environ.get("GB_CELL_LL_REP_TOL", "0.05"))
        bt = st.get("band_temps")
        if (bt is not None
                and os.environ.get("GB_CELL_LL_TEMP_SCALED", "1") == "1"):
            beta = xp.clip(
                xp.asarray(bt)[b_i, t_i].astype(float), 1e-8, 1.0)
            allowed = tol * xp.maximum(1.0, 1.0 / beta)
        else:
            allowed = xp.full(rate.shape, tol)
        excess = rate / allowed
        k = int(excess.argmax())
        e_k = float(excess[k])
        if e_k > st["max_excess"]:
            st["max_excess"] = e_k
            st["worst"] = (
                int(t_i[k]), int(w_i[k]), int(b_i[k]), int(nrep[k]),
                float(mm[k]), float(rate[k]), float(allowed[k]),
            )
        st["max_mm"] = max(st["max_mm"], float(xp.abs(mm).max()))
        st["sum_abs_mm"] += float(xp.abs(mm).sum())
        st["n_done"] += int(len(slots))
        st["open"][slots] = False

    def _cell_ll_report(self, st):
        """One line per unit: sampled-vs-realized stats over finished cells."""
        if st["n_done"] == 0 or st["worst"] is None:
            return
        t_i, w_i, b_i, nrep, mm, rate, allowed = st["worst"]
        logger.info(
            f"[GB_CELL_LL {self.name}] unit: {st['n_done']} cells credited"
            f" from the buffer before/after ll; |sampled-actual| mean"
            f" {st['sum_abs_mm'] / st['n_done']:.3e} max {st['max_mm']:.3e};"
            f" worst per-repeat {rate:.3e}/rep vs allowance {allowed:.3e}"
            f" (temp {t_i}, walker {w_i}, band {b_i}, {nrep} reps,"
            f" diff {mm:.3e})."
        )
        if rate > allowed:
            logger.warning(
                f"[GB_CELL_LL {self.name}] per-repeat sampled-vs-actual diff"
                f" {rate:.3e} exceeds its temperature-scaled allowance"
                f" {allowed:.3e} (temp {t_i}, walker {w_i}, band {b_i},"
                f" {nrep} repeats): the sampled lls and the realized buffer"
                " residual disagree beyond the expected floor for this rung."
            )

    def _run_ortho_premise_check(self, model, band_sorter, units, remainder):
        """[GB_ORTHO] orthogonality premise check (default OFF).

        PHYSICS RULING (user, verified premise): FD inner product ~0
        implies WDM inner product ~0, EVEN within one wavelet layer, so
        the concurrency constraint for GB sub-bands is ORTHOGONALITY
        (frequency separation), not disjoint wavelet-pixel support. Two
        sources with ``|df| * Tobs >> 1`` have ``<h_i|h_j> ~ 0`` and, by
        bilinearity, additive likelihood deltas -- which is what lets
        same-unit cells run concurrently. Boundary pairs (sources near a
        shared sub-band edge, small ``df``) are the one place the premise
        weakens, so this check MEASURES them: at unit close it samples the
        unit's closest-frequency cross-band cold-chain pairs
        (:func:`_ortho_boundary_pairs`) and evaluates each pair's
        normalized overlap ``|<h_i|h_j>| / sqrt(<h_i|h_i> <h_j|h_j>)``
        through the INSTALLED swap-likelihood kernels
        (``BandLikelihoodEngine.get_swap_ll`` -> ``hh_cross`` /
        ``hh_add`` / ``hh_remove``; the same inner products production
        scoring uses -- never hand-rolled).

        Knobs: ``GB_ORTHO_CHECK=1`` enables; ``GB_ORTHO_TOL`` (default
        1e-3) is the WARN threshold on the max overlap;
        ``GB_ORTHO_MAX_PAIRS`` (default 8) caps the pairs per unit.
        Diagnostic only -- never mutates state, and any internal failure
        logs a warning instead of breaking the sampler.
        """
        if os.environ.get("GB_ORTHO_CHECK", "0") != "1":
            return
        try:
            alive = _to_numpy(band_sorter.inds).astype(bool)
            t_i = _to_numpy(band_sorter.temp_inds)
            w_i = _to_numpy(band_sorter.walker_inds)
            b_i = _to_numpy(band_sorter.band_inds)
            f0 = _to_numpy(band_sorter.coords_in[:, 1])  # physical f0 (Hz)
            max_pairs = int(os.environ.get("GB_ORTHO_MAX_PAIRS", "8"))
            i_idx, j_idx = _ortho_boundary_pairs(
                f0, w_i, b_i, alive & (t_i == 0), units, remainder,
                max_pairs=max_pairs,
            )
            if i_idx.size == 0:
                logger.info(
                    "[GB_ORTHO %s] unit (bands %% %d == %d): no cold-chain "
                    "cross-band boundary pairs to check.",
                    self.name, units, remainder,
                )
                return
            xp = self.xp
            rows_i = xp.asarray(i_idx)
            rows_j = xp.asarray(j_idx)
            params_i = band_sorter.coords_in[rows_i]
            params_j = band_sorter.coords_in[rows_j]
            data_index = xp.asarray(w_i[i_idx]).astype(xp.int32)
            N_vals = xp.maximum(
                band_sorter.N_vals[rows_i], band_sorter.N_vals[rows_j]
            )
            res = self._likelihood_engine.get_swap_ll(
                model.analysis_container_arr,
                params_j,
                params_i,
                data_index=data_index,
                noise_index=data_index,
                N_vals=N_vals,
                phase_maximize=False,
                waveform_kwargs=self.waveform_kwargs,
            )
            hh_i = np.abs(_to_numpy(res.hh_add))
            hh_j = np.abs(_to_numpy(res.hh_remove))
            hh_x = np.abs(_to_numpy(res.hh_cross))
            norm = np.sqrt(np.maximum(hh_i * hh_j, 1e-300))
            overlap = hh_x / norm
            k = int(overlap.argmax())
            df_pairs = f0[j_idx] - f0[i_idx]
            tol = float(os.environ.get("GB_ORTHO_TOL", "1e-3"))
            logger.info(
                "[GB_ORTHO %s] unit (bands %% %d == %d): %d boundary pairs; "
                "normalized overlap |<h_i|h_j>|/sqrt(<h_i|h_i><h_j|h_j>) "
                "mean %.3e max %.3e (walker %d, bands %d/%d, df %.3e Hz).",
                self.name, units, remainder, int(overlap.size),
                float(overlap.mean()), float(overlap[k]),
                int(w_i[i_idx[k]]), int(b_i[i_idx[k]]), int(b_i[j_idx[k]]),
                float(df_pairs[k]),
            )
            if float(overlap[k]) > tol:
                logger.warning(
                    "[GB_ORTHO %s] max boundary-pair overlap %.3e exceeds "
                    "GB_ORTHO_TOL=%.1e (walker %d, bands %d/%d, df %.3e "
                    "Hz): the orthogonality premise is weak for this pair "
                    "-- consider a larger GB_BAND_UNIT_STRIDE or wider "
                    "minimum bands.",
                    self.name, float(overlap[k]), tol,
                    int(w_i[i_idx[k]]), int(b_i[i_idx[k]]),
                    int(b_i[j_idx[k]]), float(df_pairs[k]),
                )
        except Exception as exc:  # diagnostic only: never break the sampler
            logger.warning(
                "[GB_ORTHO %s] premise check skipped: %r", self.name, exc
            )

    def _run_band_unit(self, model, band_sorter, subset, band_temps,
                       ll_change_log, prop_counts, acc_counts):
        """Drive one parity unit's cells through the sub-band buffer."""
        tm = getattr(self, "_prop_timer", None)
        _sched_specials = subset.special_band_inds
        _cap_m = getattr(self, "_rj_at_cap_mask", None)
        if (
            _cap_m is not None
            and self.is_rj_prop
            and not self.rj_removal_only
            and not self.rj_replace
            and os.environ.get("GB_RJ_LIVE_CAP_PICK", "1") == "1"
        ):
            # Countable-row scheduler init (cap-transition invariant,
            # user design 2026-08-14): the finish budget counts alive rows
            # plus dead rows of cells BELOW cap at unit open. At-cap
            # cells' staged birth reserve is excluded here and
            # promoted/demoted by BandScheduler.add_counts at live cap
            # transitions (see _run_rj_step), so every cell finishes —
            # and retires — exactly when its live-pickable work is done.
            # Every staged cell keeps >= 1 countable row (at-cap cells
            # hold >= cap alive rows), so the cell set is unchanged.
            _countable = subset.inds | ~_cap_m[subset.inds_main_band_sorter]
            _sched_specials = subset.special_band_inds[_countable]
        scheduler = BandScheduler(
            _sched_specials, self.num_band_preload_total, xp=self.xp
        )
        with _tspan(tm, "buffer_build"):
            buffer_obj = self._cached_get_buffer(
                subset, model.analysis_container_arr,
                scheduler.slot_specials.copy(),
            )
        if tm is not None:
            tm.count("cells", int(scheduler.n_cells))
        # F-stat distance-center HOIST (2026-08-14; job-187 sync autopsy:
        # the per-round center chain cost 735 s/propose = half the rj
        # black box). Computed once per unit over the unit's COUNTABLE
        # rows (2026-08-15: alive + below-cap dead; the at-cap birth
        # reserve is left to the lookup's inline fallback — see
        # _precompute_fstat_centers) and looked up per round;
        # GB_RJ_FSTAT_CTR_HOIST=0 restores the per-round computation.
        # Reset unconditionally so a unit that skips the precompute can
        # never see a previous unit's cache.
        #
        # SUPERSEDED BY DEFAULT (GB_FSTAT_CTR_MODE=epoch, user ruling
        # 2026-08-15): with a live epoch center table the unit precompute is
        # skipped entirely -- rows look the centers up by f0 from the table
        # built once at fit time (_fstat_ctr_table_lookup). This branch is
        # the "unit" escape hatch (and the automatic fallback for a move
        # with no table).
        self._fstat_ctr = None
        if (
            self.rj_fstat_dist_birth
            and not self.rj_replace
            and os.environ.get("GB_RJ_FSTAT_CTR_HOIST", "1") == "1"
            and self._fstat_ctr_table_active() is None
        ):
            with _tspan(tm, "rj_fstat_centers"):
                self._fstat_ctr = self._precompute_fstat_centers(
                    model, band_sorter, subset
                )
        # Intra-propose memory telemetry: a grouped-RJ propose can run for
        # hours across many units with no lifecycle line until exit, which
        # left the 2026-08-13 OOM (pool 11->40 GB inside one propose)
        # unattributable. Device-wide numbers too: the pool only sees the
        # CURRENT device's cupy arrays -- the 96 GB H100 that OOM'd held
        # ~50 GB the pool stats never showed (other device's pool, raw
        # C++/CUDA allocations). GB_UNIT_POOL_LOG_EVERY=0 disables.
        if self.backend.uses_cupy:
            n_units = self._unit_open_count = getattr(
                self, "_unit_open_count", 0) + 1
            _every = int(os.environ.get("GB_UNIT_POOL_LOG_EVERY", "25"))
            if _every > 0 and n_units % _every == 0:
                logger.info(
                    "%s: unit %d open (%d cells): GPU pool used "
                    "%.2f / total %.2f GB; %s", self.name, n_units,
                    int(scheduler.n_cells),
                    self.mempool.used_bytes() / 1e9,
                    self.mempool.total_bytes() / 1e9,
                    self._device_mem_summary())
        self._debug_log_band_null(buffer_obj)

        # Buffer before/after cell ll crediting (2026-08-12 user design):
        # a cell's stay in a buffer slot is the smallest running unit --
        # fill -> ll -> rj -> in-model repeats -> ll -> diff at switch-out.
        # The diff replaces the cell's accumulated sampled lls in
        # ll_change_log (see _cell_ll_finalize). GB_CELL_LL_CREDIT=0
        # restores the pure sampled-ll ledger.
        cell_ll_state = None
        if os.environ.get("GB_CELL_LL_CREDIT", "1") == "1":
            cell_ll_state = self._cell_ll_state_init(
                scheduler, band_temps=band_temps)
            _slots0 = scheduler.xp.arange(scheduler.n_slots)[
                scheduler.slot_active
            ]
            with _tspan(tm, "cell_ll"):
                self._cell_ll_open(
                    cell_ll_state, buffer_obj, _slots0,
                    scheduler.slot_specials[_slots0],
                    ll_change_log, prop_counts,
                )

        # Pick eligibility lives on the MAIN sorter: only sources inside this
        # unit's subset are candidates (for in-model moves the subset already
        # applied ``inds``; for RJ it includes the freshly-drawn dead ones).
        eligible = self.xp.zeros(band_sorter.num_sources, dtype=bool)
        eligible[subset.inds_main_band_sorter] = True
        # Unit-scoped eligibility, consumed by _run_rj_step's cap-transition
        # budget adjustment (counting a freed/re-capped cell's UNPICKED
        # staged birth rows requires knowing which main-sorter rows belong
        # to this unit).
        self._unit_eligible = eligible

        def _advance_and_refill(frozen_specials=None):
            """Retire finished cells and refill their slots (with the cell-ll
            credit bracketing the switch-out). Returns the number of slots
            refilled. Cells named in ``frozen_specials`` (grouped RJ pool)
            are never retired: a pending source pins its cell's buffer slot
            until the in-model flush runs."""
            inds_fill, new_specials = scheduler.advance(
                frozen_specials=frozen_specials
            )
            if len(inds_fill):
                # Switch-out: credit the outgoing cells from their slab
                # before/after ll BEFORE the refill overwrites the slots.
                if cell_ll_state is not None:
                    with _tspan(tm, "cell_ll"):
                        self._cell_ll_finalize(
                            cell_ll_state, buffer_obj, inds_fill,
                            ll_change_log, prop_counts,
                        )
                with _tspan(tm, "buffer_build"):
                    subset.get_buffer(
                        model.analysis_container_arr, new_specials,
                        inds_fill=inds_fill, buffer_obj=buffer_obj,
                        timer=tm,
                    )
                self._debug_log_band_null(buffer_obj)
                if cell_ll_state is not None:
                    with _tspan(tm, "cell_ll"):
                        self._cell_ll_open(
                            cell_ll_state, buffer_obj, inds_fill,
                            new_specials, ll_change_log, prop_counts,
                        )
            return int(len(inds_fill))

        def _free_mempool_each_round():
            # GPU efficiency (parallel-resources plan P1): freeing the WHOLE
            # CuPy pool every pick round forces cudaFree/cudaMalloc churn
            # for every allocation in the next round, so it is now OPT-IN —
            # set GB_MEMPOOL_FREE_EACH_ROUND=1 only when a run is genuinely
            # memory-bound (the per-unit/per-proposal frees remain). The
            # mempool_free stage time quantifies the cost either way.
            if os.environ.get("GB_MEMPOOL_FREE_EACH_ROUND", "0") == "1":
                with _tspan(tm, "mempool_free"):
                    self.mempool.free_all_blocks()

        # GROUPED RJ -> in-model scheduling (2026-08-13 user design,
        # GB_RJ_GROUPED_INMODEL=0 restores the per-round interleave): RJ
        # rounds (one proposal per cell per round, exactly as before) run
        # back-to-back, ACCUMULATING every source that ends the round alive
        # (accepted birth, or survived/skipped death) into a pending pool —
        # its cell is then FROZEN so the pool never holds two same-cell
        # sources. When no unfrozen cell has candidates left, the scheduler
        # first STAGES NEW CELLS into every non-frozen finished slot and the
        # RJ sweep continues (full-width flush rule, 2026-08-14): the pool
        # keeps growing across staging cohorts until nothing more can load,
        # and only then does ONE in-model block — as close to buffer-width
        # as the unit's survivor count allows — evolve the whole pool
        # together; the pool clears, and the
        # sweep continues over the remaining sources. Same proposals, same
        # statistics — the in-model repeat loop just always runs at full
        # batch width instead of on each round's (often tiny) survivor set.
        # At-cap cells: dead slots were excluded from ``eligible`` up front
        # (no birth RJ at all); alive slots still get death proposals but
        # NEVER pool/freeze (user rule 2026-08-13) — only below-cap cells
        # add sources via RJ and then freeze them for the polish flush.
        grouped = (
            self.is_rj_prop
            and os.environ.get("GB_RJ_GROUPED_INMODEL", "1") == "1"
        )

        round_i = 0
        # DIRECT-BATCH RJ -> in-model (user design 2026-08-14,
        # GB_RJ_DIRECT_BATCH=0 restores the staged-scheduler path): the
        # dynamic scheduler (per-cell retirement, continuous refill, pool
        # freezing, finish budgets) is replaced by RIGID BATCHES — bind
        # the buffer to the next batch of cells, run RJ pick rounds until
        # they exhaust, pool the survivors, move on; when every batch has
        # finished its RJ, ONE in-model phase polishes all the unit's
        # survivors in capacity-width chunks. The fixed, repeating
        # fill -> rounds -> accept shape per batch is what CUDA graph
        # capture needs. Batch termination is simply "_pick_sources
        # returned None": live-cap-filtered rows never pick, a freed
        # cell's rows become pickable in later rounds of the SAME batch
        # (re-entry), and there is no budget bookkeeping to go stale —
        # TODO (user, 2026-08-14): SCRUTINIZE the never-freed tail — if a
        # cell stays at cap for the whole batch, its unvisited (invisible,
        # live-cap-filtered) birth rows are simply never picked and the
        # batch ends without them. That is the intended semantics (same
        # net effect as the old unit-open exclusion, minus the one-unit
        # wait for freed cells), but verify in production that (a) the
        # batch-end round census confirms those rows never leak into
        # picks, (b) cells freed in a LATER batch of the same unit do NOT
        # regain their rows (they were bound in the earlier batch — a
        # cross-batch re-entry gap the scheduler path did not have
        # either), and (c) the [GB_ACCEPT rj-split] "capped" class stays
        # ~0 so no at-cap birth reaches the kernel through this path.
        # no freeze, no advance, no deadlock surface. Survivor slot
        # indices are recomputed at in-model bind time (RJ-time slots are
        # stale after rebinds); ONE pooled survivor per cell (host-side
        # dedup) replaces the freeze as the serial-within-band guarantee.
        direct = grouped and os.environ.get("GB_RJ_DIRECT_BATCH", "1") == "1"
        if direct:
            xp = self.xp

            class _BatchView:
                """Scheduler stand-in: the batch IS the active set.

                No retirement / refill / freezing / finish budgets —
                ``add_counts`` (the live-cap transition budget) is a no-op
                because nothing retires by budget here; cap re-entry works
                through the live pick filter alone.
                """

                def __init__(self, specials):
                    self.active_slot_specials = specials

                def add_counts(self, specials, deltas):
                    return

            pending = []
            _pooled_host = set()
            all_cells = scheduler.cell_specials
            n_slots = int(scheduler.n_slots)
            n_cells_total = int(scheduler.n_cells)
            n_batches = max(1, -(-n_cells_total // max(n_slots, 1)))

            def _rebind(specials_new):
                """Finalize the outgoing binding's cell-ll credit, rebind
                THE buffer to the new cell set, open credit brackets."""
                nonlocal buffer_obj
                if cell_ll_state is not None:
                    _open_now = scheduler.xp.arange(scheduler.n_slots)[
                        cell_ll_state["open"]
                    ]
                    with _tspan(tm, "cell_ll"):
                        self._cell_ll_finalize(
                            cell_ll_state, buffer_obj, _open_now,
                            ll_change_log, prop_counts,
                        )
                with _tspan(tm, "buffer_build"):
                    buffer_obj = self._cached_get_buffer(
                        subset, model.analysis_container_arr,
                        specials_new.copy(),
                    )
                if cell_ll_state is not None:
                    with _tspan(tm, "cell_ll"):
                        self._cell_ll_open(
                            cell_ll_state, buffer_obj,
                            xp.arange(int(len(specials_new))),
                            specials_new, ll_change_log, prop_counts,
                        )
                return buffer_obj

            for _b in range(n_batches):
                batch_specials = all_cells[_b * n_slots:(_b + 1) * n_slots]
                if _b > 0:
                    buffer_obj = _rebind(batch_specials)
                bview = _BatchView(batch_specials)
                while True:
                    with _tspan(tm, "pick"):
                        picked = self._pick_sources(
                            band_sorter, buffer_obj, bview, eligible,
                        )
                    if picked is None:
                        break
                    if tm is not None:
                        tm.count(
                            "picked_sources", int(len(picked["specials"])))
                    # Pick-time provenance for the per-class in-model
                    # budgets (user ruling 2026-08-15): a row DEAD here
                    # that is alive after the RJ step is an accepted birth
                    # ("newborn"); a row ALIVE here is a mature survivor.
                    # Captured on the CALLER's pre-flip-gate dict (the RJ
                    # step's own flip-fraction subset never re-aligns with
                    # it). Device-resident -- no host sync.
                    alive_at_pick = band_sorter.inds[picked["ids"]].copy()
                    rj_seq = self._debug_rj_select(buffer_obj, picked)
                    with _tspan(tm, "rj_step"):
                        if self.rj_replace:
                            self._run_replace_step(
                                model, band_sorter, buffer_obj, band_temps,
                                picked, ll_change_log, prop_counts,
                                acc_counts, round_i, bview,
                            )
                        else:
                            self._run_rj_step(
                                model, band_sorter, buffer_obj, band_temps,
                                picked, ll_change_log, prop_counts,
                                acc_counts, round_i, bview,
                            )
                    self._debug_plot_rj_pair(buffer_obj, rj_seq)
                    # Survivor pooling: PRE-accept cap mask (newborns pool;
                    # at-cap death-rejected survivors don't — user rules
                    # 2026-08-13/14), then host-side one-per-cell dedup.
                    alive_now = band_sorter.inds[picked["ids"]]
                    _lcs = getattr(self, "_live_cap_state", None)
                    if _lcs is not None:
                        _counts_pre, _cap_arr = _lcs
                        # Pool gate on the CAP-CELL grid (2026-08-15): an
                        # at-cap CELL never freezes its sources into the
                        # in-model pool.
                        _p_flat = self._cap_flat_index(
                            picked["temp_inds"], picked["walker_inds"],
                            picked["cap_inds"],
                        )
                        alive_now = alive_now & ~(
                            _counts_pre[_p_flat]
                            >= _cap_arr[picked["cap_inds"]]
                        )
                    else:
                        _at_cap_m = getattr(self, "_rj_at_cap_mask", None)
                        if _at_cap_m is not None:
                            alive_now = alive_now & ~_at_cap_m[picked["ids"]]
                    if bool(alive_now.any()):
                        held = {k: v[alive_now] for k, v in picked.items()}
                        # Provenance rides with the pooled rows: True =
                        # accepted birth (dead at pick), False = mature.
                        # Removal-only / replace steps never revive a dead
                        # row, so their pools are 100% mature here by
                        # construction.
                        held["newborn"] = (~alive_at_pick)[alive_now]
                        _sp_h = np.asarray(_to_numpy(held["specials"]))
                        _keep = np.fromiter(
                            (s not in _pooled_host for s in _sp_h.tolist()),
                            dtype=bool, count=len(_sp_h),
                        )
                        if _keep.any():
                            _pooled_host.update(_sp_h[_keep].tolist())
                            _km = xp.asarray(_keep)
                            pending.append(
                                {k: v[_km] for k, v in held.items()})
                    round_i += 1
                    _free_mempool_each_round()

            # IN-MODEL PHASE: every batch's RJ is done — polish ALL the
            # unit's survivors in capacity-width chunks (the full-width
            # rule holds by construction: only the final chunk can be
            # narrower, because fewer survivors remain).
            n_chunks = 0
            n_surv = 0
            _cls_census = {"newborn": 0, "mature": 0}
            _cls_reps = {
                "newborn": self.inmodel_repeats_newborn,
                "mature": self.inmodel_repeats_survivor,
            }
            if pending:
                merged = (
                    pending[0] if len(pending) == 1 else {
                        k: xp.concatenate([p[k] for p in pending])
                        for k in pending[0]
                    }
                )
                n_surv = int(len(merged["specials"]))
                # In-model chunk width cap (2026-08-14, job-194 forensics):
                # a progressive leaf-cap rise (1 -> 2) unlocked the removal
                # move's first big survivor pool and the single 7,532-source
                # in-model block that followed built a sig-het in-model
                # reference large enough to push a 96 GB device to ~84%
                # (the sig-het stash is single-device) -- the restart-era
                # OOM signature. Healthy history is blocks <= ~2,300
                # sources; cap the chunk width independently of the buffer
                # capacity so reference size stays bounded as caps rise.
                _im_w = min(
                    n_slots,
                    max(1, int(os.environ.get("GB_RJ_INMODEL_CHUNK", "4096"))),
                )
                # PER-CLASS chunk sequences (user ruling 2026-08-15):
                # newborns and mature survivors run separate fixed repeat
                # budgets (search 200 / 25, PE 100 / 100 stock), so the
                # pool splits by pick-time provenance first and each class
                # then chunks under the same width cap. The full-width rule
                # relaxes to per-class: only each class's FINAL chunk can
                # be narrower than the cap. Cell disjointness holds across
                # classes (the host-side dedup above is class-blind).
                for _cls_name, _cls in _split_by_newborn(merged, xp):
                    _cls_census[_cls_name] = int(len(_cls["specials"]))
                    for _st in range(0, _cls_census[_cls_name], _im_w):
                        chunk = {
                            k: v[_st:_st + _im_w] for k, v in _cls.items()
                        }
                        buffer_obj = _rebind(chunk["specials"])
                        # RJ-time slot indices are stale after the rebind.
                        chunk["slot_index"] = buffer_obj.get_index(
                            chunk["specials"]).astype(xp.int32)
                        with _tspan(tm, "inmodel_repeats"):
                            self._run_in_model_repeats(
                                model, band_sorter, buffer_obj, band_temps,
                                chunk, ll_change_log, prop_counts,
                                acc_counts,
                                num_repeats=_cls_reps[_cls_name],
                            )
                        n_chunks += 1
            logger.info(
                f"{self.name}: direct batches — {n_batches} rj batch(es), "
                f"{n_surv} survivors polished in {n_chunks} in-model "
                f"chunk(s) ({n_slots} buffer slots; "
                f"newborn {_cls_census['newborn']}@{_cls_reps['newborn']} "
                f"/ mature {_cls_census['mature']}@{_cls_reps['mature']})."
            )
            if tm is not None:
                tm.count("inmodel_flushes", n_chunks)
        elif grouped:
            pending = []
            pending_specials = self.xp.zeros(
                0, dtype=band_sorter.special_band_inds.dtype
            )
            n_flushes = 0
            flush_sum = 0
            while scheduler.any_active():
                with _tspan(tm, "pick"):
                    picked = self._pick_sources(
                        band_sorter, buffer_obj, scheduler, eligible,
                        blocked_specials=pending_specials,
                    )
                if picked is not None:
                    if tm is not None:
                        tm.count("picked_sources", int(len(picked["specials"])))
                    rj_seq = self._debug_rj_select(buffer_obj, picked)
                    with _tspan(tm, "rj_step"):
                        if self.rj_replace:
                            self._run_replace_step(
                                model, band_sorter, buffer_obj, band_temps,
                                picked, ll_change_log, prop_counts,
                                acc_counts, round_i, scheduler,
                            )
                        else:
                            self._run_rj_step(
                                model, band_sorter, buffer_obj, band_temps,
                                picked, ll_change_log, prop_counts,
                                acc_counts, round_i, scheduler,
                            )
                    self._debug_plot_rj_pair(buffer_obj, rj_seq)
                    scheduler.record_picks(picked["specials"])
                    # Post-RJ alive sources join the pool; their cells
                    # freeze until the flush. AT-CAP cells never pool (user
                    # rule 2026-08-13): only below-cap cells add sources via
                    # RJ and then freeze them for the in-model flush — an
                    # at-cap cell keeps proposing deaths round after round
                    # but its survivors skip the polish block. Newborns
                    # always pool: a birth requires a below-cap cell at unit
                    # open (dead slots of at-cap cells never enter the pick
                    # pool), and the mask is that same unit-open snapshot.
                    alive_now = band_sorter.inds[picked["ids"]]
                    # Pool gate on the PRE-accept mask (user ruling
                    # 2026-08-14): "at-cap cells never pool" is judged at
                    # the cell state WHEN THE ROW WAS PROPOSED — a newborn
                    # (its cell was below cap at proposal) pools for its
                    # in-model repeats even though the accept just put the
                    # cell at cap; a death-rejected survivor of an at-cap
                    # cell does not. Post-accept evaluation would block
                    # every newborn in cap-1 bands and gut the grouped
                    # in-model. Live state comes from the round's
                    # _pick_sources stash; fallback = unit-open snapshot.
                    _lcs = getattr(self, "_live_cap_state", None)
                    if _lcs is not None:
                        _counts_pre, _cap_arr = _lcs
                        _p_flat = self._cap_flat_index(
                            picked["temp_inds"], picked["walker_inds"],
                            picked["cap_inds"],
                        )
                        _pre_cap_row = (
                            _counts_pre[_p_flat]
                            >= _cap_arr[picked["cap_inds"]]
                        )
                        alive_now = alive_now & ~_pre_cap_row
                    else:
                        _at_cap_m = getattr(self, "_rj_at_cap_mask", None)
                        if _at_cap_m is not None:
                            alive_now = alive_now & ~_at_cap_m[picked["ids"]]
                    if bool(alive_now.any()):
                        held = {k: v[alive_now] for k, v in picked.items()}
                        pending.append(held)
                        pending_specials = self.xp.concatenate(
                            [pending_specials, held["specials"]]
                        )
                    round_i += 1
                    _free_mempool_each_round()
                    continue

                # No pickable cell outside the frozen set. FULL-WIDTH FLUSH
                # RULE (user design 2026-08-14): the in-model block must not
                # run below the buffer width unless fewer sources remain, so
                # FIRST try to stage more cells — frozen cells' slots stay
                # pinned (advance skips them), every other finished slot
                # refills, and the RJ sweep continues pooling into the new
                # cohort. Only when nothing more can stage does the pool
                # flush through ONE in-model block; a narrower-than-buffer
                # flush therefore only happens when the unit genuinely has
                # fewer poolable survivors than slots.
                n_refilled = _advance_and_refill(
                    frozen_specials=(
                        pending_specials if len(pending_specials) else None
                    )
                )
                if n_refilled:
                    _free_mempool_each_round()
                    continue
                n_flushed = 0
                if pending:
                    merged = (
                        pending[0]
                        if len(pending) == 1
                        else {
                            k: self.xp.concatenate([p[k] for p in pending])
                            for k in pending[0]
                        }
                    )
                    n_flushed = int(len(merged["specials"]))
                    if int(self.xp.unique(merged["specials"]).size) != n_flushed:
                        # Serial-within-band invariant: the freeze logic
                        # guarantees one pooled source per cell.
                        raise RuntimeError(
                            f"{self.name}: grouped RJ pool holds duplicate "
                            "(temp, walker, band) cells — same-band sources "
                            "must never share an in-model block."
                        )
                    with _tspan(tm, "inmodel_repeats"):
                        # Scheduler (non-direct) grouped path: ONE repeat
                        # count for the whole pool = the survivor/mature
                        # budget (per-class partitioning lives on the
                        # direct-batch path only; noted user trade-off).
                        self._run_in_model_repeats(
                            model, band_sorter, buffer_obj, band_temps,
                            merged, ll_change_log, prop_counts, acc_counts,
                            num_repeats=self.inmodel_repeats_survivor,
                        )
                    n_flushes += 1
                    flush_sum += n_flushed
                    pending = []
                    pending_specials = pending_specials[:0]
                n_refilled = _advance_and_refill()
                _free_mempool_each_round()
                if n_flushed == 0 and n_refilled == 0:
                    # Nothing pooled and nothing new to load: every active
                    # cell is exhausted (advance() just deactivated them).
                    break
            if n_flushes:
                logger.info(
                    f"{self.name}: grouped in-model — {n_flushes} flushes, "
                    f"mean batch {flush_sum / n_flushes:.1f} sources "
                    f"({scheduler.n_slots} buffer slots)."
                )
            if tm is not None:
                tm.count("inmodel_flushes", n_flushes)
        else:
            while scheduler.any_active():
                with _tspan(tm, "pick"):
                    picked = self._pick_sources(band_sorter, buffer_obj, scheduler, eligible)
                if picked is None:
                    break
                if tm is not None:
                    # Batch size per repeat round: on GPU, small batches mean
                    # the 100-repeat in-model loop is
                    # kernel-launch-overhead-bound.
                    tm.count("picked_sources", int(len(picked["specials"])))

                if self.is_rj_prop:
                    # RJ before/after trace of the chosen cell: snapshots
                    # bracket the RJ step; figures save only when the cell's
                    # RJ proposal was ACCEPTED (buffer changed).
                    # Chronologically BEFORE the in-model sequence figures.
                    rj_seq = self._debug_rj_select(buffer_obj, picked)
                    with _tspan(tm, "rj_step"):
                        if self.rj_replace:
                            # Fixed-dimension replacement instead of
                            # birth/death.
                            self._run_replace_step(
                                model, band_sorter, buffer_obj, band_temps,
                                picked, ll_change_log, prop_counts, acc_counts,
                                round_i, scheduler,
                            )
                        else:
                            self._run_rj_step(
                                model, band_sorter, buffer_obj, band_temps, picked,
                                ll_change_log, prop_counts, acc_counts, round_i, scheduler,
                            )
                    self._debug_plot_rj_pair(buffer_obj, rj_seq)

                with _tspan(tm, "inmodel_repeats"):
                    # Non-grouped per-round interleave: RJ moves use the
                    # single survivor/mature budget; pure in-model moves
                    # (is_rj_prop=False -- the search "in_model" move,
                    # VGB) keep the plain ``num_repeat_proposals``.
                    self._run_in_model_repeats(
                        model, band_sorter, buffer_obj, band_temps, picked,
                        ll_change_log, prop_counts, acc_counts,
                        num_repeats=(
                            self.inmodel_repeats_survivor
                            if self.is_rj_prop else None
                        ),
                    )

                scheduler.record_picks(picked["specials"])
                _advance_and_refill()
                round_i += 1
                _free_mempool_each_round()
        if cell_ll_state is not None:
            # Unit end: cells still resident (active or retired-in-place)
            # never hit a refill -- finalize them against the final slab.
            _open_slots = scheduler.xp.arange(scheduler.n_slots)[
                cell_ll_state["open"]
            ]
            with _tspan(tm, "cell_ll"):
                self._cell_ll_finalize(
                    cell_ll_state, buffer_obj, _open_slots,
                    ll_change_log, prop_counts,
                )
            self._cell_ll_report(cell_ll_state)

        if tm is not None:
            tm.count("pick_rounds", round_i)

        logger.info(
            f"{self.name}: band unit complete after {round_i} pick rounds "
            f"({scheduler.n_cells} cells)."
        )

    def _pick_sources(self, band_sorter, buffer_obj, scheduler, eligible,
                      blocked_specials=None):
        """One not-yet-visited source per active cell, without replacement.

        Vectorized on ``self.xp``: candidates are gathered through the
        special-index maps, randomly ranked within each cell, and the first
        per cell wins. ``band_sorter.has_run_rj`` marks consumed sources for
        the remainder of this proposal, so every source is visited exactly
        once per pass.

        ``blocked_specials`` (grouped RJ scheduling) removes whole cells from
        the candidate pool: a cell already holding a pending alive source is
        frozen until the accumulated in-model flush runs, so the pool can
        never collect two same-cell sources (serial-within-band rule).

        LIVE at-cap birth gate (user decision table 2026-08-14,
        ``GB_RJ_LIVE_CAP_PICK=0`` disables): as a row comes up —
        ``inds=False`` and its cell at capacity RIGHT NOW → thrown away
        (never a candidate: no round budget, no counters); ``inds=False``
        below capacity → birth attempt; ``inds=True`` → death attempt
        always. Recomputed from live ``band_sorter.inds`` every round, so
        a cell freed by an accepted death regains its births the next
        round and loses them again if re-capped — any number of
        transitions. The pre-accept per-cell counts are stashed on
        ``self._live_cap_state`` for the same round's transition budget
        adjustment and the in-model pool gate (BOTH must see the
        pre-accept state).
        """
        xp = self.xp
        cand = (
            eligible
            & (~band_sorter.has_run_rj)
            & band_sorter.get_subset_bool(
                special_band_inds=scheduler.active_slot_specials
            )
        )
        if blocked_specials is not None and len(blocked_specials):
            cand = cand & ~xp.isin(
                band_sorter.special_band_inds, blocked_specials
            )
        self._live_cap_state = None
        if (
            self.is_rj_prop
            and not self.rj_removal_only
            and not self.rj_replace
            and self._cap_leaf_cap is not None
            and os.environ.get("GB_RJ_LIVE_CAP_PICK", "1") == "1"
        ):
            # Live census on the CAP-CELL grid (2026-08-15). Dead rows are
            # gated on whether their whole SUB-BAND is saturated (a birth
            # can land in any of its cells); alive rows on their own cell.
            _cap_inds_all = self._sorter_cap_cells(band_sorter)
            flat_all, _counts = self._cap_cell_counts(
                band_sorter, _cap_inds_all
            )
            _cap = xp.asarray(self._cap_leaf_cap)
            self._live_cap_state = (_counts, _cap)
            _at_cap_row = self._cap_at_cap_mask(
                band_sorter, _counts, _cap, flat_all, _cap_inds_all
            )
            cand = cand & (band_sorter.inds | ~_at_cap_row)
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
            # Cap cell of the row AS IT STANDS. Meaningful for alive rows
            # (deaths / in-model); a BIRTH's cell is recomputed from the
            # drawn frequency at the prior gate in _run_rj_step.
            "cap_inds": self._cap_cell_index(
                band_inds, band_sorter.freqs[ids]
            ),
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
        # Under GB_SIGHET_INMODEL=1 ``gb_wdm_comp`` is a
        # GBSignalHetComputations wrapper, which forwards only the band-engine
        # surface (fill_global_wdm / get_ll_wdm / get_swap_ll_wdm / grads /
        # information_matrix) to its chunked delegate -- it has no
        # __getattr__, so get_fstat_ll_wdm is not reachable through it. Unwrap
        # to the delegate: the F-stat is scored against the parent ACA residual
        # passed explicitly below, never against the in-model heterodyne
        # reference, so the chunked delegate is the correct target whether or
        # not a sig-het reference is currently active.
        wdm_comp = getattr(self.gb_wdm_comp, "chunked", self.gb_wdm_comp)
        # The router takes the comp OBJECT plus the entry-point NAME (not a
        # bound method) so it can resolve the shard's device-local replica
        # before binding.
        comp, method_name = (
            (self.gb_fd_comp, "get_fstat_ll_fd")
            if isinstance(self._basis_settings, FDSettings)
            else (wdm_comp, "get_fstat_ll_wdm")
        )
        return _RoutedBandEngine.route_fstat_ll(
            comp, method_name, holder, params_phys,
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

    # SNR-truncation boundary floor in standardized units: when the analytic
    # boundary would leave essentially no lognormal mass (ln_dist_max <=
    # ln_center - 6*sigma), alpha clamps here so Phi(alpha) >= ~1e-9 and the
    # -log Phi(alpha) normalization stays finite (~ +20.7). The CLAMPED alpha
    # feeds the draw AND both density sides identically, so the proposal is
    # still an exactly-normalized truncated lognormal — its boundary is just
    # held at the -6 sigma tail — and detailed balance is unaffected. Births
    # from such degenerate rows land in the deepest allowed tail and die at
    # the actual opt_snr clamp exactly as before.
    _SNR_TRUNC_ALPHA_FLOOR = -6.0

    def _snr_trunc_alpha(self, ln_snr, sigma, snr_limit):
        """Standardized SNR-truncation boundary ``alpha``, floored.

        ``alpha = ln(snr_center / snr_limit) / sigma``: since opt SNR
        scales as ``1/dist`` at fixed intrinsics and the center amplitude's
        F-stat SNR is ``exp(ln_snr)``, the ``opt_snr >= snr_limit`` region
        is ``ln dist <= ln_center + sigma * alpha`` (distance basis) /
        ``lnA >= ln_center - sigma * alpha`` (amplitude basis). ``sigma``
        is the SMEARED proposal width actually used for the draw, so alpha
        is exact in the standardized draw coordinate.
        """
        xp = self.xp
        alpha = (ln_snr - float(np.log(snr_limit))) / sigma
        return xp.clip(alpha, self._SNR_TRUNC_ALPHA_FLOOR, None)

    def _std_norm_cdf(self, x):
        """Standard normal CDF ``Phi`` on ``self.xp`` arrays."""
        xp = self.xp
        if xp is np:
            from scipy.special import erf
        else:
            from cupyx.scipy.special import erf
        return 0.5 * (1.0 + erf(xp.asarray(x) / np.sqrt(2.0)))

    def _std_norm_ppf(self, p):
        """Standard normal inverse CDF ``Phi^-1`` on ``self.xp`` arrays."""
        xp = self.xp
        if xp is np:
            from scipy.special import erfinv
        else:
            from cupyx.scipy.special import erfinv
        return np.sqrt(2.0) * erfinv(2.0 * xp.asarray(p) - 1.0)

    def _truncnorm_std_draw(self, n, alpha):
        """SNR-truncated standardized draw — ONE uniform per row.

        Inverse-CDF sampling: ``z = Phi^-1(U * Phi(alpha))`` is a standard
        normal truncated ABOVE at ``alpha``. The distance basis uses it
        directly (large z = large dist = small SNR); the amplitude basis
        negates it (lower truncation at ``-alpha``: small lnA = small SNR).
        RNG consumption is one generator call of ``n`` values per round —
        the same shape as the untruncated ``cp.random.randn(n)`` it
        replaces (the raw stream VALUES differ, so runs with the knob
        flipped are not draw-for-draw comparable;
        ``GB_RJ_SNR_TRUNC_DIST=0`` restores the randn path bit-identically).
        For ``alpha >> 1``, ``Phi(alpha) == 1.0`` in double precision and
        the draw reduces to plain inverse-CDF standard-normal sampling.
        """
        xp = self.xp
        u = xp.asarray(cp.random.rand(n))
        p = u * self._std_norm_cdf(alpha)
        # U in [0, 1) can hit 0 exactly; keep the ppf argument inside (0, 1).
        p = xp.clip(p, 1e-15, 1.0 - 1e-16)
        z = self._std_norm_ppf(p)
        return z if _gb_use_distance(self) else -z

    def _slot0_log_proposal(self, slot0_vals, ln_center, sigma, alpha=None):
        """``log g`` of the slot-0 value under the F-stat lognormal proposal.

        The proposal is Gaussian in the LOG of slot 0 (log-distance or lnA),
        so the density in the sampled coordinate ``v`` is
        ``g(v) = N(ln v; ln_center, sigma) / v`` (the ``1/v`` is the
        ``ln v -> v`` Jacobian). For the amplitude basis slot 0 is ALREADY lnA
        (sampled in log space), so there is no Jacobian term there.

        ``alpha`` (optional): standardized SNR-truncation boundary from
        :meth:`_snr_trunc_alpha` (already floored). When given, the density
        is the TRUNCATED lognormal — ``-log Phi(alpha)`` renormalizes it
        (per row, since alpha varies per row), and values outside the
        support (beyond the max-distance / min-amplitude boundary, with a
        1e-10-standardized slack absorbing the exp/log FP round trip of
        stored draws) get log-density -1e300. Death-side calls pass the
        SAME per-row alpha as the birth side, so detailed balance is
        exact; the consequence is that a death whose slot-0 value lies
        outside the support is force-rejected HERE (the truncated birth
        proposal could never have produced that value — the prior-RJ death
        path remains the removal route for such rows). For ``alpha >> 1``
        ``Phi(alpha) == 1.0`` in double precision and the support test
        never fires, so the truncated density equals the untruncated one
        exactly.
        """
        xp = self.xp
        lv = xp.log(xp.clip(slot0_vals, 1e-300, None)) if _gb_use_distance(self) else slot0_vals
        logg = (
            -0.5 * ((lv - ln_center) / sigma) ** 2
            - xp.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        )
        if _gb_use_distance(self):
            logg = logg - lv  # Jacobian d(ln dist)/d(dist) = 1/dist
        if alpha is not None:
            logg = logg - xp.log(self._std_norm_cdf(alpha))
            u = (lv - ln_center) / sigma
            side = u if _gb_use_distance(self) else -u
            logg = xp.where(side > alpha + 1e-10, -1e300, logg)
        return logg

    def _temper_cadence_fire(self) -> bool:
        """Tempering cadence gate (user design 2026-08-14).

        Fires when at least ``temper_every_proposes`` TOTAL branch
        proposes (shared census across every move instance) have elapsed
        since the branch last tempered, and records the firing. Called
        LAST in the swap-stage condition chain so it only consumes the
        budget when every other gate already passed. n <= 1 is the
        legacy every-eligible-propose behavior. On a fresh process the
        first eligible propose always fires (ladder adaptation starts
        immediately after any restart).
        """
        n = getattr(self, "temper_every_proposes", 1)
        if n <= 1:
            return True
        cnt = GBSpecialBase._branch_propose_counts.get(self.branch_name, 0)
        last = GBSpecialBase._branch_last_temper.get(self.branch_name)
        if last is not None and (cnt - last) < n:
            return False
        GBSpecialBase._branch_last_temper[self.branch_name] = cnt
        return True

    def _band_shutoff_enabled(self) -> bool:
        """High-frequency band birth shutoff: is it on for this move?

        USER KEY CHANGE 2026-08-15: the criterion is OCCUPANCY, not
        acceptance (the acceptance version never fired — hot-chain churn
        kept every band's accept tally nonzero). See
        :meth:`_update_band_shutoff` for the occupancy rules. Bands with
        LOWER edge above ``GB_RJ_BAND_SHUTOFF_FMIN_MHZ`` (default 10 mHz
        — no confusion noise up there, so a real source shows full SNR
        from the first iteration) are eligible. RJ BIRTHS ONLY are shut
        off: deaths and in-model repeats continue for any resident
        source. ``GB_RJ_BAND_SHUTOFF_SCOPE``: "search" (default) = only
        moves with "search" in their name; "all" = every fstat birth
        move; "off" = disabled. Exactly ONE enabled move should exist
        per stage (the fstat birth move) — a second would double-count
        iterations. 6-mo TODO: reassess search+pe vs search-only.
        Counters are in-memory per move instance — a restart resets
        them and bands re-earn their shutoff (errs conservative).
        """
        if (not self.is_rj_prop or self.rj_removal_only or self.rj_replace
                or not getattr(self, "rj_fstat_dist_birth", False)):
            return False
        scope = os.environ.get("GB_RJ_BAND_SHUTOFF_SCOPE", "search")
        if scope == "off":
            return False
        if scope == "search" and "search" not in self.name:
            return False
        return True

    def _update_band_shutoff(self, occ_max) -> None:
        """Occupancy-based shutoff update (USER KEY CHANGE 2026-08-15).

        ``occ_max``: host int array (num_bands,) — COLD-CHAIN occupancy
        per band, MAX over walkers, measured once per iteration at
        propose end of the (single) designated move. Rules, exactly as
        ruled:

        - occupancy == 0 for ``GB_RJ_BAND_SHUTOFF_AFTER`` (default 5)
          consecutive iterations -> births OFF (nothing ever sticks).
        - occupancy == 1 for AFTER consecutive iterations, where the
          band's leaf cap was > 1 THROUGHOUT the streak -> births OFF
          (a second source was ALLOWED the whole time and never
          arrived). Iterations at cap <= 1 RESET the streak rather
          than pause it — the band never had the chance to add a
          second, so they cannot count (conservative).
        - occupancy >= 2, or ANY occupancy change, resets the streak
          (a fresh first source restarts the one-source clock; a
          death 2 -> 1 starts a fresh one-source streak).

        Shutoff is permanent for the process (restart re-earns;
        revival semantics deliberately not implemented — an OFF band
        whose source later dies stays OFF). Emits the LOG CONTRACT
        prefix the monitor's cap-plot overlay parses
        (``[GB_BAND_SHUTOFF <move>] band <b> ...``).
        """
        occ_max = np.asarray(occ_max)
        if not hasattr(self, "_band_occ_streak"):
            self._band_occ_streak = np.zeros(self.num_bands, dtype=np.int64)
            self._band_occ_last = np.full(self.num_bands, -1, dtype=np.int64)
            self._rj_band_shutoff = np.zeros(self.num_bands, dtype=bool)
        fmin_mhz = float(os.environ.get("GB_RJ_BAND_SHUTOFF_FMIN_MHZ", "10.0"))
        after = int(os.environ.get("GB_RJ_BAND_SHUTOFF_AFTER", "5"))
        # Band SHUTOFF stays a per-BAND rule (user design 2026-08-15: only
        # the caps move to the cell grid). It asks "could this band hold a
        # second source anywhere", so the per-band allowance is the MAX over
        # the band's cap cells -- which is exactly what
        # ``_mirror_band_leaf_cap`` keeps ``band_leaf_cap`` equal to.
        cap = getattr(self, "_band_leaf_cap", None)
        if cap is None:
            # No cap machinery -> a second source is always allowed.
            cap_h = np.full(self.num_bands, np.iinfo(np.int64).max)
        else:
            cap_h = np.asarray(_to_numpy(cap))
        qualifying = (occ_max == 0) | ((occ_max == 1) & (cap_h > 1))
        unchanged = occ_max == self._band_occ_last
        self._band_occ_streak = np.where(
            qualifying & unchanged, self._band_occ_streak + 1,
            np.where(qualifying, 1, 0),
        )
        self._band_occ_last = occ_max.copy()
        edges = _to_numpy(self.band_edges)
        hi_f = edges[:-1] * 1e3 >= fmin_mhz
        new_off = hi_f & ~self._rj_band_shutoff & (
            self._band_occ_streak >= after)
        for b in np.where(new_off)[0]:
            self._rj_band_shutoff[b] = True
            logger.info(
                "[GB_BAND_SHUTOFF %s] band %d (%.3f-%.3f mHz) births OFF "
                "after %d iterations at occupancy %d (%d bands off)",
                self.name, int(b), edges[b] * 1e3, edges[b + 1] * 1e3,
                int(self._band_occ_streak[b]), int(occ_max[b]),
                int(self._rj_band_shutoff.sum()))

    def _band_occupancy_cold_max(self, state) -> np.ndarray:
        """Cold-chain per-band occupancy, MAX over walkers (host array).

        Leaves are binned by their OWN f0 (searchsorted on band_edges,
        same convention as the BandSorter); alive leaves outside the
        band range are ignored.
        """
        work_b = self._work_branch(state)
        inds0 = np.asarray(_to_numpy(work_b.inds[0]), dtype=bool)
        f0_hz = np.asarray(_to_numpy(work_b.coords[0, :, :, 1])) / 1e3
        edges = _to_numpy(self.band_edges)
        b = np.searchsorted(edges, f0_hz) - 1
        valid = inds0 & (b >= 0) & (b < self.num_bands)
        occ = np.zeros((self.nwalkers, self.num_bands), dtype=np.int64)
        w_idx, l_idx = np.nonzero(valid)
        np.add.at(occ, (w_idx, b[valid]), 1)
        return occ.max(axis=0)

    # Field names of the per-row F-stat center cache, in the order
    # _fstat_ctr_compute returns them (the lookup-miss fallback appends
    # to every one of these in lockstep with "ids").
    _FSTAT_CTR_FIELDS = ("phi0", "iota", "psi", "ln_center", "sigma", "ln_snr")

    @staticmethod
    def _fstat_ctr_mode() -> str:
        """Which F-stat center machinery runs: ``"epoch"`` (default) or
        ``"unit"``.

        USER RULING 2026-08-15: *"just compute the center distributions 1
        time when we build the fstat distribution in the first place (inside
        setup()). Compute them once, smear them out for inaccuracy issues
        and call it a day."*

        * ``epoch`` -- ONE batched sweep over the birth proposal's drawable
          f0 support at fit time (:meth:`GBSpecialRJFStatGridMove._install`),
          persisted in the epoch cache dir and looked up per row by f0
          (:meth:`_fstat_ctr_table_lookup`). No per-unit precompute, no
          lookup-miss fallback, NO per-row F-stat evaluation at propose time
          — the 109-953 s/propose center chain becomes a searchsorted.
        * ``unit`` -- the escape hatch: the per-unit countable-row hoist
          (:meth:`_precompute_fstat_centers`) exactly as before, bit-identical
          when selected.

        ``epoch`` silently degrades to ``unit`` for any move that has no
        table (an RJ fstat-birth move that never fits a grid, or an epoch
        whose support was empty) — see :meth:`_fstat_ctr_table_active`.
        """
        mode = os.environ.get("GB_FSTAT_CTR_MODE", "epoch").strip().lower()
        if mode not in ("epoch", "unit"):
            raise ValueError(
                f"GB_FSTAT_CTR_MODE must be 'epoch' or 'unit', got {mode!r}")
        return mode

    def _fstat_ctr_smear(self) -> float:
        """Multiplicative widening of the center lognormal's ``sigma``.

        ``GB_FSTAT_CTR_SMEAR`` always wins. Otherwise the default is
        MODE-DEPENDENT, because the two modes carry different staleness:

        * ``unit`` -> 1.5: the cache reads the walker-ref residual once per
          parity unit, so it covers mid-unit residual drift only.
        * ``epoch`` -> 2.0: the table is built once per fit epoch, so it
          covers up to ``GB_FSTAT_REFIT_EVERY`` proposes of residual drift
          PLUS the node-vs-row f0/Mc/sky mismatch of the nearest-node lookup.

        The smeared sigma feeds the draw AND both density sides identically
        (:meth:`_slot0_log_proposal`), so the proposal stays an exactly
        normalized (truncated) lognormal — just broader — and detailed
        balance is untouched.
        """
        raw = os.environ.get("GB_FSTAT_CTR_SMEAR", "").strip()
        if raw:
            return float(raw)
        return 2.0 if self._fstat_ctr_mode() == "epoch" else 1.5

    def _fstat_ctr_table_active(self):
        """The live epoch center table, or ``None`` when it must not be used.

        ``None`` means "fall back to the unit hoist / per-round compute":
        either the mode knob selects ``unit``, or this move has no table
        (it never fitted a grid, or the epoch had no drawable f0 support).
        """
        if self._fstat_ctr_mode() != "epoch":
            return None
        return getattr(self, "_fstat_ctr_table", None)

    def _fstat_ctr_table_lookup(self, rows_params):
        """Per-row centers from the epoch table — NEAREST node in f0.

        ``rows_params`` are sampling-basis rows (column 1 = f0 [mHz],
        column 2 = Mc), the SAME array the per-round F-stat path would have
        been handed; births look up at their drawn f0 and deaths at the
        leaf's current f0, so the pair is symmetric by construction.

        NEAREST, not interpolated. The stored tuple ``(phi0, iota, psi,
        A_max, F)`` is one joint F-stat maximum: adjacent nodes can sit on
        DIFFERENT sources (node spacing is ~the matched-filter peak width),
        and phi0/iota/psi are angles, so component-wise linear interpolation
        would blend incompatible maxima and wrap-average angles. Nearest
        keeps every row on a physically realizable maximum and pushes the
        node-vs-row mismatch entirely into the smear.

        Only the F-stat MAXIMUM comes from the table. ``(ln_center, sigma)``
        are re-derived per row through the shared
        :meth:`_dist_center_and_width` with the row's OWN ``(f0, Mc)``, so
        the distance basis' ``ln dist* = ln(amp_from_dist(f0, Mc, 1)/A_max)``
        keeps its exact per-row ``Mc**(5/3)`` scaling (that term spans ~11
        e-folds across the Mc prior — far beyond anything a smear could
        cover). Returns the ``_FSTAT_CTR_FIELDS`` 6-tuple.
        """
        xp = self.xp
        t = self._fstat_ctr_table
        f0_nodes = t["f0_mHz"]
        f0 = xp.asarray(rows_params[:, 1], dtype=xp.float64)
        n = int(f0_nodes.shape[0])
        hi = xp.clip(xp.searchsorted(f0_nodes, f0), 0, n - 1)
        lo = xp.clip(hi - 1, 0, n - 1)
        take_lo = xp.abs(f0 - f0_nodes[lo]) <= xp.abs(f0_nodes[hi] - f0)
        pos = xp.where(take_lo, lo, hi)
        ln_snr = t["ln_snr"][pos]
        A_max = xp.exp(t["ln_A_max"][pos])
        # clip(2F, 1) == exp(2 ln_snr) by construction, and only the clipped
        # combination enters _dist_center_and_width -> sigma == sigma_base.
        F = 0.5 * xp.exp(2.0 * ln_snr)
        ln_center, sigma = self._dist_center_and_width(rows_params, A_max, F)
        sigma = sigma * self._fstat_ctr_smear()
        return (t["phi0"][pos], t["iota"][pos], t["psi"][pos],
                ln_center, sigma, ln_snr)

    def _fstat_ctr_compute(self, model, params):
        """Batched F-stat center computation for a set of rows.

        The shared low-level path for the unit-open precompute
        (:meth:`_precompute_fstat_centers`) AND the per-round lookup-miss
        fallback (:meth:`_fstat_ctr_lookup`) — both MUST produce identical
        numbers for the same rows, so the ``GB_FSTAT_CTR_BATCH`` batching
        (default 4096, the comb sweep's proven batch), the Jaranowski-Krol
        inversion, the ``(ln_center, sigma)`` mapping and the
        ``GB_FSTAT_CTR_SMEAR`` widening all live here.

        Returns ``(phi0, iota, psi, ln_center, sigma, ln_snr)`` on
        ``self.xp``. ``ln_snr = ln sqrt(max(2F, 1))`` is the F-stat SNR at
        the center amplitude ``A_max`` (the same clipped ``2F`` that sets
        the pre-smear ``sigma = 1/snr`` in :meth:`_dist_center_and_width`;
        cached explicitly because the smear makes ``sigma`` unusable for
        recovering it). It is the quantity the SNR-truncated distance
        proposal turns into its analytic boundary: opt SNR scales as
        ``1/dist`` at fixed intrinsics, so ``SNR >= limit`` <=>
        ``ln dist <= ln_center + ln(snr_center/limit)`` (amplitude basis:
        ``lnA >= ln_center - ln(snr_center/limit)``).
        """
        xp = self.xp
        n = int(params.shape[0])
        walker_ref = getattr(self, "_fstat_walker_ref", 0)
        batch = int(os.environ.get("GB_FSTAT_CTR_BATCH", "4096"))
        A = xp.zeros(n)
        phi0 = xp.zeros(n)
        iota = xp.zeros(n)
        psi = xp.zeros(n)
        F = xp.zeros(n)
        for st in range(0, n, batch):
            en = min(st + batch, n)
            (A[st:en], phi0[st:en], iota[st:en], psi[st:en],
             F[st:en]) = self._fstat_dist_centers(
                model, params[st:en], walker_ref)
        ln_center, sigma = self._dist_center_and_width(params, A, F)
        # Snapshot-drift smear (user ruling 2026-08-14): the cache reads the
        # live walker-ref residual ONCE at unit open, so the lognormal is
        # WIDENED by _fstat_ctr_smear (unit-mode default 1.5x) to cover mid-unit
        # residual drift. The smeared sigma feeds BOTH the draw and the
        # forward/reverse densities, so the proposal remains exactly
        # detailed-balance-valid -- just broader. (A search-mode variant
        # that re-centers without paying the density -- deliberately
        # breaking detailed balance for faster burn-in -- was considered
        # and PARKED, user 2026-08-14: "maybe not ideal".)
        sigma = sigma * self._fstat_ctr_smear()
        ln_snr = 0.5 * xp.log(xp.clip(2.0 * F, 1.0, None))
        return phi0, iota, psi, ln_center, sigma, ln_snr

    def _precompute_fstat_centers(self, model, band_sorter, subset):
        """Unit-open F-stat center cache for the distance-birth proposal.

        The job-187 sync autopsy measured the per-round center chain at
        735 s/propose — half the rj black box — for math whose inputs are
        all fixed at unit open: birth coordinates are pre-drawn at sorter
        build, an alive row's coordinates cannot change before its single
        RJ pick (``has_run_rj``; in-model updates only touch rows AFTER
        they pool), and the parent residual is in exactly the state the
        first pick round would see (the parity class is opened before
        :meth:`_run_band_unit`). So the (A, phi0, iota, psi, F) maxima and
        the slot-0 ``(ln_center, sigma)`` are computed ONCE here, batched
        through the F-stat comp (:meth:`_fstat_ctr_compute`), and looked up
        per round. Mid-unit drift of the reference walker's residual (its
        own accepted flips) is the same order of approximation the
        per-round path already accepted mid-propose — and the cache is at
        least internally consistent across the unit where the per-round
        path drifted.

        COUNTABLE-ONLY precompute (2026-08-15; job-195 measured
        rj_fstat_centers at ~372 s/propose with ~80% of the hoist spent on
        AT-CAP birth RESERVE rows at cap 1-2): only rows that can actually
        consume a center this unit are precomputed —

        - ALIVE rows. NOT skippable: every death pick evaluates the
          REVERSE-proposal density at its own center
          (``_run_rj_step``'s death block), so alive rows consume centers.
        - DEAD rows of cells BELOW cap at unit open (pickable births).
          Dead rows of AT-CAP cells never reach a center lookup while the
          cell stays capped: under live-cap gating they are never picked,
          and under the ``GB_RJ_LIVE_CAP_PICK=0`` regimes they are either
          excluded from the subset (``GB_RJ_SKIP_CAPPED=1``) or -inf'ed at
          the prior gate before ``birth_k`` is formed.

        THE TRAP this pairs with (same-commit user constraint): under
        ``GB_RJ_LIVE_CAP_PICK=1`` leaf caps change MID-propose — a cell
        freed by an accepted death exposes its reserve rows to the pick
        pool, and the cache lacks them. :meth:`_fstat_ctr_lookup` therefore
        carries an inline per-round fallback: a miss on a row that belongs
        to this unit (``unit_ids``) computes the missing centers through
        the SAME :meth:`_fstat_ctr_compute` path (batched over that round's
        misses) and appends them, so a row misses at most once; the
        snapshot smear covers the mid-unit residual skew exactly as it does
        for the precomputed rows. A miss on a row OUTSIDE the unit is still
        the loud stale-cache RuntimeError.
        """
        xp = self.xp
        ids = subset.inds_main_band_sorter
        if int(len(ids)) == 0:
            return None
        # Full unit membership (ascending), kept for the lookup fallback's
        # reserve-row vs foreign-id distinction.
        unit_ids = xp.asarray(ids)
        cap_m = getattr(self, "_rj_at_cap_mask", None)
        if cap_m is not None:
            # subset.inds is the per-row alive bool aligned with
            # inds_main_band_sorter — same countable arithmetic as
            # _run_band_unit's scheduler-budget init.
            countable = subset.inds | ~cap_m[ids]
            ids = ids[countable]
        params = band_sorter.coords[ids]
        _t0 = time.perf_counter()
        phi0, iota, psi, ln_center, sigma, ln_snr = self._fstat_ctr_compute(
            model, params)
        # Per-unit precompute census (2026-08-15, job-195 diagnostic: the
        # production rj_fstat_centers stage jumped 374 -> 1953 s/propose on
        # identical code with caps/cells/rounds flat -- this line pins
        # whether the ROW POPULATION or the PER-ROW F-stat cost grew).
        _n_unit = int(len(unit_ids))
        _n_rows = int(len(ids))
        _n_alive = int(subset.inds.sum()) if _n_unit else 0
        logger.info(
            "[FSTAT_CTR %s] unit precompute: %d rows (%d alive / %d "
            "countable-birth; %d at-cap excluded of %d unit rows) in %.2fs; "
            "propose fallback rows so far %d",
            self.name, _n_rows, _n_alive, _n_rows - _n_alive,
            _n_unit - _n_rows, _n_unit, time.perf_counter() - _t0,
            int(getattr(self, "_fstat_ctr_fallback_rows", 0)),
        )
        return {
            # ids is ascending by construction (arange[bool] in
            # get_subset_inds; the countable mask preserves that order) --
            # _fstat_ctr_lookup relies on it.
            "ids": xp.asarray(ids),
            "unit_ids": unit_ids,
            "phi0": phi0, "iota": iota, "psi": psi,
            "ln_center": ln_center, "sigma": sigma, "ln_snr": ln_snr,
            "n_miss": 0,
        }

    def _fstat_ctr_lookup(self, rows_ids, model=None, band_sorter=None):
        """Cache positions of main-sorter ``rows_ids`` (verified gather).

        Countable-only cache (see :meth:`_precompute_fstat_centers`): a
        miss on a row that belongs to this unit (``unit_ids``) is a
        live-cap reserve row exposed mid-unit — its centers are computed
        inline through the SAME :meth:`_fstat_ctr_compute` path (batched
        over the round's misses) and APPENDED to the cache, so each row
        misses at most once. A miss on a row outside the unit means the
        cache is stale/foreign — fail loudly, the factors it would produce
        are silently wrong.
        """
        xp = self.xp
        c = self._fstat_ctr
        rows_ids = xp.asarray(rows_ids)
        n_cache = int(c["ids"].shape[0])
        pos = xp.searchsorted(c["ids"], rows_ids)
        if n_cache:
            # Clip the gather: searchsorted returns n_cache for ids beyond
            # the last cached id (a miss, not an index error).
            hit = c["ids"][xp.minimum(pos, n_cache - 1)] == rows_ids
        else:
            hit = xp.zeros(rows_ids.shape, dtype=bool)
        if bool(hit.all()):
            return pos
        miss_ids = xp.unique(rows_ids[~hit])
        unit_ids = c.get("unit_ids")
        in_unit = xp.zeros(miss_ids.shape, dtype=bool)
        if unit_ids is not None and int(unit_ids.shape[0]):
            upos = xp.minimum(
                xp.searchsorted(unit_ids, miss_ids),
                int(unit_ids.shape[0]) - 1,
            )
            in_unit = unit_ids[upos] == miss_ids
        if not bool(in_unit.all()):
            raise RuntimeError(
                f"{self.name}: F-stat center cache does not cover the "
                "picked rows and they are outside the unit's subset "
                "(stale unit cache?)"
            )
        if model is None or band_sorter is None:
            raise RuntimeError(
                f"{self.name}: F-stat center cache miss on in-unit reserve "
                "rows but the caller supplied no model/band_sorter for the "
                "inline fallback"
            )
        new_vals = self._fstat_ctr_compute(model, band_sorter.coords[miss_ids])
        new_ids = xp.concatenate([c["ids"], miss_ids])
        order = xp.argsort(new_ids)
        c["ids"] = new_ids[order]
        for name, vals in zip(self._FSTAT_CTR_FIELDS, new_vals):
            c[name] = xp.concatenate([c[name], vals])[order]
        c["n_miss"] = int(c.get("n_miss", 0)) + int(miss_ids.shape[0])
        self._fstat_ctr_fallback_rows = (
            int(getattr(self, "_fstat_ctr_fallback_rows", 0))
            + int(miss_ids.shape[0])
        )
        logger.debug(
            "%s: F-stat center cache fallback computed %d reserve rows "
            "(unit total misses %d)", self.name, int(miss_ids.shape[0]),
            c["n_miss"])
        return xp.searchsorted(c["ids"], rows_ids)

    def _apply_rj_flip_fraction(self, band_sorter, picked):
        """Gate the DEATH attempts of picked ALIVE rows to the flip subset.

        Births are thinned EARLY (unit-open subset exclusion, 2026-08-14
        user design): every dead row that reaches a pick already belongs to
        the flip subset, so re-gating it here would square the fraction —
        births therefore pass unconditionally. Alive rows are always
        pickable (they must pool for their in-model repeats regardless of
        the flip; user rule 2026-08-12), so their death attempt is thinned
        HERE instead: a gated alive row drops out of the RJ step while the
        caller's pre-gate ``picked`` dict still pools it. The mask is drawn
        ONCE per proposal (attached to the per-propose ``band_sorter``,
        same lifetime as ``has_run_rj``). Returns the filtered ``picked``
        dict, or ``None`` when nothing survives. Fraction 1.0 is a
        pass-through (bit-identical to the historical behavior).
        """
        if self.rj_flip_fraction >= 1.0:
            return picked
        xp = self.xp
        allowed = getattr(band_sorter, "_rj_flip_allowed", None)
        if allowed is None:
            n = int(band_sorter.num_sources)
            n_keep = max(1, int(round(self.rj_flip_fraction * n)))
            allowed = xp.zeros(n, dtype=bool)
            allowed[xp.random.permutation(n)[:n_keep]] = True
            band_sorter._rj_flip_allowed = allowed
        keep = allowed[picked["ids"]] | ~band_sorter.inds[picked["ids"]]
        if not bool(keep.any()):
            return None
        if bool(keep.all()):
            return picked
        return {key: value[keep] for key, value in picked.items()}

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

        ``rj_flip_fraction`` < 1: births are thinned EARLY (unit-open
        subset exclusion — gated dead rows never enter the pick pool), so
        only the DEATH attempts of picked alive rows are gated here (see
        :meth:`_apply_rj_flip_fraction`); the in-model repeats that follow
        are NOT restricted (gated alive rows still pool).
        """
        picked = self._apply_rj_flip_fraction(band_sorter, picked)
        if picked is None:
            return
        # Checkpoint timing INSIDE the rj step (the 85%-of-propose black
        # box): host wall between marks. Launches are async, so by default
        # device work lands on whichever mark forces the next sync;
        # GB_PROP_TIMING_SYNC=1 syncs at EVERY mark so each stage carries
        # exactly its own kernel time (same contract as _ProposeTimer.span).
        _tm_rj = getattr(self, "_prop_timer", None)
        _rj_sync = getattr(_tm_rj, "_sync", None)
        if _rj_sync is not None:
            _rj_sync()
        _t_mark = time.perf_counter()

        def _mark(_name):
            nonlocal _t_mark
            if _tm_rj is None:
                return
            if _rj_sync is not None:
                _rj_sync()
            _now = time.perf_counter()
            _tm_rj.add(_name, _now - _t_mark)
            _t_mark = _now

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

        # Removal-only mode (search pruning heuristic; see the ctor
        # docstring): force-reject every birth row -- the -inf routes it
        # through the existing ``keep`` machinery, so it never reaches the
        # likelihood kernel and the bad-accept guard rejects it at beta > 0.
        # Deaths run untouched; their factors are this instance's
        # ``rj_proposal_distribution`` logpdf as usual (a prior-removal
        # instance = GBSpecialRJPriorMove with the PRIOR container).
        if self.rj_removal_only:
            curr_logp[~alive] = -np.inf

        # Births outside this cell's frequency window are unphysical.
        f_hz = params[:, 1] / 1e3
        out_of_band = (
            (f_hz < buffer_obj.frequency_lims[0][slots])
            | (f_hz > buffer_obj.frequency_lims[1][slots])
        )
        curr_logp[(~alive) & out_of_band] = -np.inf
        # [GB_ACCEPT rj-split] class stashes (denominator split, user
        # request 2026-08-14): which gate killed each birth before/at
        # scoring. Populated below; consumed just before the accept mark.
        _split_over_cap = None
        _split_snr = None
        _split_kernel_rows = None
        # Cap cells of the picked rows AT THE PRIOR GATE (drawn frequency
        # for births); reused by the accept block's cap-transition budget.
        _gate_cap_cells = None

        # Per-band progressive leaf cap (search mode): a birth into a band
        # already holding ``cap[b]`` alive sources is prior-forbidden --
        # a truncation of the prior on the per-band leaf count. The cap is
        # judged on the cold chain (``_update_band_leaf_caps``) but enforced
        # at EVERY temperature so tempering swaps stay within a common prior
        # support. Setting -inf here routes the birth through the existing
        # ``keep`` machinery: it never reaches the likelihood kernel and the
        # bad-accept guard force-rejects it at beta > 0.
        if self._cap_leaf_cap is not None:
            # Reuse THIS round's live-cap census when _pick_sources built
            # one (GB_RJ_LIVE_CAP_PICK path): ``band_sorter.inds`` cannot
            # change between the pick and this gate (only the accept block
            # below flips it), so the (counts, cap) pair is IDENTICAL to
            # the recompute it replaces -- one full-sorter flat-index build
            # + bincount saved per pick round. ``_live_cap_state`` is reset
            # at the top of every _pick_sources call, so it can never be
            # stale across rounds.
            _lcs_gate = getattr(self, "_live_cap_state", None)
            if _lcs_gate is not None:
                cell_counts, cap_xp = _lcs_gate
            else:
                cap_xp = xp.asarray(self._cap_leaf_cap)
                _, cell_counts = self._cap_cell_counts(band_sorter)
            # THE EXACT PER-CELL ENFORCEMENT POINT (2026-08-15). A birth's
            # cap cell is set by its DRAWN frequency, not by the dead
            # slot's stale coords -- the draw covers the whole sub-band, so
            # any of its cells is reachable. The pick-side gates upstream
            # are throughput heuristics on band saturation; THIS is the
            # correctness backstop, and it is per cell.
            _f0_prop = band_sorter.coords_freqs_hz(params)
            if _f0_prop is None:
                cap_cells_gate = picked["cap_inds"]
            else:
                cap_cells_gate = self._cap_cell_index(
                    picked["band_inds"], _f0_prop
                )
            cell_flat = self._cap_flat_index(
                picked["temp_inds"], picked["walker_inds"], cap_cells_gate
            )
            over_cap = (
                cell_counts[cell_flat] >= cap_xp[cap_cells_gate]
            )
            _split_over_cap = (~alive) & over_cap
            curr_logp[(~alive) & over_cap] = -np.inf
            # The accept block's cap-transition budget needs the SAME cells.
            _gate_cap_cells = cap_cells_gate

        _mark("rj_prior_gate")
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
                # Unit-open center cache (see _precompute_fstat_centers);
                # None = hoist disabled -> per-round computation below.
                # Countable-only cache: the lookup carries the inline
                # reserve-row fallback, so it needs model/band_sorter.
                _ctr = getattr(self, "_fstat_ctr", None)
                # Epoch center TABLE (default; user ruling 2026-08-15): built
                # once per fit epoch over the proposal's drawable f0 support
                # and looked up by f0 -- no per-unit precompute, no fallback
                # machinery, no per-row F-stat eval on either side. Deaths
                # read the SAME table at their current f0, so the forward and
                # reverse densities stay exactly symmetric.
                _tbl = self._fstat_ctr_table_active()
                # SNR-truncated distance proposal (2026-08-15, user-ruled
                # lever; GB_RJ_SNR_TRUNC_DIST=0 restores the untruncated
                # lognormal draw bit-identically): [GB_ACCEPT rj-split]
                # showed 54-62% of scored births dying at the
                # opt_snr < limit clamp. The center's F-stat SNR (ln_snr,
                # from the same cache/fallback interface as the centers)
                # makes the clamp boundary analytic in the draw coordinate
                # -- alpha = ln(snr_center/limit)/sigma -- so the draw is
                # truncated there and the truncated density (including its
                # per-row -log Phi(alpha) normalization) replaces the
                # untruncated one on BOTH the birth and the death side of
                # the RJ factors: detailed balance stays exact.
                _snr_lim = float(buffer_obj.opt_snr_rej_samp_limit)
                _snr_trunc = (
                    os.environ.get("GB_RJ_SNR_TRUNC_DIST", "1") == "1"
                    and _snr_lim > 0.0
                )
                if len(birth_k):
                    if _tbl is not None:
                        (phi0_max, iota_max, psi_max, ln_center, sigma,
                         ln_snr_b) = self._fstat_ctr_table_lookup(
                            params[birth_k])
                    elif _ctr is not None:
                        _bpos = self._fstat_ctr_lookup(
                            ids[birth_k], model=model,
                            band_sorter=band_sorter)
                        phi0_max = _ctr["phi0"][_bpos]
                        iota_max = _ctr["iota"][_bpos]
                        psi_max = _ctr["psi"][_bpos]
                        ln_center = _ctr["ln_center"][_bpos]
                        sigma = _ctr["sigma"][_bpos]
                        ln_snr_b = _ctr["ln_snr"][_bpos]
                    else:
                        A_max, phi0_max, iota_max, psi_max, F = self._fstat_dist_centers(
                            model, params[birth_k], walker_ref)
                        ln_center, sigma = self._dist_center_and_width(
                            params[birth_k], A_max, F)
                        ln_snr_b = 0.5 * xp.log(xp.clip(2.0 * F, 1.0, None))
                    if _snr_trunc:
                        alpha_b = self._snr_trunc_alpha(
                            ln_snr_b, sigma, _snr_lim)
                        z = self._truncnorm_std_draw(len(birth_k), alpha_b)
                    else:
                        alpha_b = None
                        z = xp.asarray(cp.random.randn(len(birth_k)))
                    ln_draw = ln_center + sigma * z
                    if _gb_use_distance(self):
                        params[birth_k, 0] = xp.exp(ln_draw)
                    else:
                        params[birth_k, 0] = ln_draw  # slot 0 is lnA already
                    params[birth_k, 4] = xp.cos(iota_max % np.pi)
                    params[birth_k, 5] = psi_max % np.pi
                    params[birth_k, 3] = phi0_max % (2 * np.pi)
                    _bl = self._slot0_log_proposal(
                        params[birth_k, 0], ln_center, sigma, alpha=alpha_b)
                    _fstat_factor_corr[birth_k] = -_bl - _log_range
                    _mark("rj_fstat_centers")
                    # Re-evaluate the global prior at the drawn distance/angles
                    # (the earlier curr_logp used the placeholder draw); f0,
                    # band and leaf-cap gating are unchanged by this overwrite.
                    curr_logp[birth_k] = cp.asarray(
                        self.gpu_priors[self.branch_name].logpdf(params[birth_k]))
                    _mark("rj_birth_prior")
                    oob_rows = _eval(birth_k, True)
                    if buffer_obj.phase_angle is not None:
                        params[birth_k, 3] = params[birth_k, 3] - buffer_obj.phase_angle
                    _mark("rj_getll")
                if len(death_k):
                    oob_rows = xp.concatenate([oob_rows, _eval(death_k, False)])
                    _mark("rj_getll")
                    if _tbl is not None:
                        (_, _, _, ln_center_d, sigma_d,
                         ln_snr_d) = self._fstat_ctr_table_lookup(
                            params[death_k])
                    elif _ctr is not None:
                        _dpos = self._fstat_ctr_lookup(
                            ids[death_k], model=model,
                            band_sorter=band_sorter)
                        ln_center_d = _ctr["ln_center"][_dpos]
                        sigma_d = _ctr["sigma"][_dpos]
                        ln_snr_d = _ctr["ln_snr"][_dpos]
                    else:
                        Ad, _pd, _id, _psd, Fd = self._fstat_dist_centers(
                            model, params[death_k], walker_ref)
                        ln_center_d, sigma_d = self._dist_center_and_width(
                            params[death_k], Ad, Fd)
                        ln_snr_d = 0.5 * xp.log(xp.clip(2.0 * Fd, 1.0, None))
                    # Reverse (birth-direction) density at the removed
                    # source's own center: the SAME per-row truncation
                    # boundary as the birth side, so the pair is exactly
                    # detailed-balance-symmetric.
                    alpha_d = (
                        self._snr_trunc_alpha(ln_snr_d, sigma_d, _snr_lim)
                        if _snr_trunc else None
                    )
                    _dl = self._slot0_log_proposal(
                        params[death_k, 0], ln_center_d, sigma_d,
                        alpha=alpha_d)
                    _fstat_factor_corr[death_k] = _dl + _log_range
                    _mark("rj_fstat_centers")
            elif self.phase_maximize and len(birth_k):
                # Maximise the birth phase; deaths keep the true phase.
                oob_rows = _eval(birth_k, True)
                if buffer_obj.phase_angle is not None:
                    params[birth_k, 3] = params[birth_k, 3] - buffer_obj.phase_angle
                if len(death_k):
                    oob_rows = xp.concatenate([oob_rows, _eval(death_k, False)])
                _mark("rj_getll")
            else:
                oob_rows = _eval(k_ids, False)
                _mark("rj_getll")

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

            # SNR prior boundary on births (search AND pe): optimal SNR
            # always; detected d_h/sqrt(h_h) only when snr_rej_detected
            # (default OFF). For amp-maximized rows d_h == h_h == snr^2,
            # so det == opt there and the extra test is a no-op anyway.
            opt_snr = xp.sqrt(xp.maximum(h_h, 0.0))
            _lim = buffer_obj.opt_snr_rej_samp_limit
            _bad_snr = opt_snr < _lim
            if getattr(buffer_obj, "snr_rej_detected", False):
                det_snr = d_h / xp.maximum(opt_snr, 1e-300)
                _bad_snr = _bad_snr | (det_snr < _lim)
            reject = (~alive) & keep & _bad_snr
            delta_ll[reject] = -1e300
            _split_snr = reject
            _split_kernel_rows = oob_rows

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

        # Logs before 2026-08-15 report everything since rj_prior_gate as one
        # ``rj_kernel`` mark; it now splits into rj_fstat_centers +
        # rj_birth_prior + rj_getll + rj_score_rest (sum the four to compare
        # against old logs). rj_score_rest = delta-ll assembly, SNR gate and
        # debug hooks after the scoring calls.
        _mark("rj_score_rest")
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
            # Look the traced source up by ID: the flip-fraction gate at the
            # top of this method re-subsets ``picked``, so the pick-time
            # positional index is stale (IndexError in smoke 3, 2026-08-12).
            # Excluded from the subset -> its proposal never ran -> rejected.
            _dbg_pos = xp.where(ids == rj_seq["source_id"])[0]
            rj_seq["accepted"] = (
                bool(accept[int(_dbg_pos[0])]) if int(_dbg_pos.size) else False
            )

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

        # [GB_ACCEPT rj-split] accumulation (one batched host transfer per
        # round): every post-flip picked row lands in exactly one class.
        # Deaths are always MH-compared; births are gated (prior / oob /
        # capped, priority in that order), scored-but-dropped (SNR clamp,
        # kernel kept_out), or VIABLE = actually MH-compared. The headline
        # "MH acceptance among viable births" is printed at propose end.
        _sp = getattr(self, "_rj_split", None)
        if _sp is not None:
            _births = ~alive
            _cold_m = t_i == 0
            _b_prior = _births & xp.isinf(logp)
            _b_oob = _births & out_of_band & ~_b_prior
            if _split_over_cap is not None:
                _b_cap = _split_over_cap & ~_b_prior & ~_b_oob
            else:
                _b_cap = xp.zeros_like(_births)
            if _split_kernel_rows is not None and len(_split_kernel_rows):
                _kr = xp.zeros(len(ids), dtype=bool)
                _kr[_split_kernel_rows] = True
                _b_kernel = _births & keep & _kr
            else:
                _b_kernel = xp.zeros_like(_births)
            _b_snr = (_split_snr if _split_snr is not None
                      else xp.zeros_like(_births))
            _b_viable = _births & keep & ~_b_kernel & ~_b_snr
            _b_acc = accept & _births
            _d_acc = accept & alive
            _vals = _to_numpy(xp.stack([
                _births.sum(), _b_prior.sum(), _b_oob.sum(),
                _b_cap.sum(), _b_snr.sum(), _b_kernel.sum(),
                _b_viable.sum(), (_b_viable & _cold_m).sum(),
                _b_acc.sum(), (_b_acc & _cold_m).sum(),
                alive.sum(), _d_acc.sum(),
            ]))
            for _kname, _v in zip(
                ("births", "prior", "oob", "capped", "snr", "kernel",
                 "viable", "viable_cold", "birth_acc", "birth_acc_cold",
                 "deaths", "death_acc"), _vals,
            ):
                _sp[_kname] = _sp.get(_kname, 0) + int(_v)

        _mark("rj_accept")
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

            # Live cap-transition budget adjustment (user design
            # 2026-08-14; invariant: cell_counts == picked + currently
            # pickable). Uses the PRE-accept per-cell alive counts stashed
            # at pick time: an accepted death at pre-count == cap FREES
            # the cell (its unpicked staged birth rows join the finish
            # budget and become pickable next round); an accepted birth
            # at pre-count == cap-1 re-CAPS it (they leave the budget
            # again). Runs synchronously with the accept — before any
            # scheduler.advance() can see a stale "finished" state.
            _lcs = getattr(self, "_live_cap_state", None)
            _uel = getattr(self, "_unit_eligible", None)
            if _lcs is not None and _uel is not None and scheduler is not None:
                _counts_pre, _cap_arr = _lcs
                _cells_acc = (
                    picked["cap_inds"] if _gate_cap_cells is None
                    else _gate_cap_cells
                )[accept]
                _flat_acc = self._cap_flat_index(
                    t_i[accept], w_i[accept], _cells_acc
                )
                _pre = _counts_pre[_flat_acc]
                _cap_acc = _cap_arr[_cells_acc]
                _alive_acc = alive[accept]
                if self.cap_divisor == 1:
                    _freed = _alive_acc & (_pre == _cap_acc)
                    _capped = (~_alive_acc) & (_pre + 1 >= _cap_acc)
                else:
                    # The scheduler's finish budget is per BAND cell and a
                    # dead row is pickable while ANY cap cell of its band
                    # still has headroom, so the budget transitions are
                    # band SATURATION transitions: a death frees the band
                    # only if the band was fully saturated before it, and a
                    # birth re-caps it only if it fills the last cell with
                    # headroom. (At divisor 1 this is exactly the branch
                    # above -- kept separate so that path stays untouched.)
                    _delta = xp.where(_alive_acc, -1, 1)
                    _counts_post = _counts_pre.copy()
                    # serial-within-band scheduling gives at most one
                    # accept per (temp, walker, band) per round, so a
                    # scatter-add is unambiguous
                    _counts_post[_flat_acc] += _delta
                    _bflat_acc = self._band_flat_index(
                        t_i[accept], w_i[accept], b_i[accept]
                    )
                    _sat_pre = self._band_saturated_flat(
                        _counts_pre, _cap_arr)[_bflat_acc]
                    _sat_post = self._band_saturated_flat(
                        _counts_post, _cap_arr)[_bflat_acc]
                    _freed = _alive_acc & _sat_pre & ~_sat_post
                    _capped = (~_alive_acc) & ~_sat_pre & _sat_post
                if bool(_freed.any()) or bool(_capped.any()):
                    _tr_specials = picked["specials"][accept]
                    _avail = (
                        _uel & ~band_sorter.has_run_rj & ~band_sorter.inds
                    )
                    _sb = band_sorter.special_band_inds
                    for _sp_arr, _sign in (
                        (_tr_specials[_freed], 1),
                        (_tr_specials[_capped], -1),
                    ):
                        if int(len(_sp_arr)) == 0:
                            continue
                        _m = _avail & xp.isin(_sb, _sp_arr)
                        _sp_sorted = xp.sort(_sp_arr)
                        if bool(_m.any()):
                            _pos = xp.searchsorted(_sp_sorted, _sb[_m])
                            _cnts = xp.bincount(
                                _pos, minlength=len(_sp_sorted))
                        else:
                            _cnts = xp.zeros(
                                len(_sp_sorted), dtype=xp.int64)
                        scheduler.add_counts(_sp_sorted, _sign * _cnts)

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
        _mark("rj_fill")

    def _run_replace_step(self, model, band_sorter, buffer_obj, band_temps,
                          picked, ll_change_log, prop_counts, acc_counts,
                          round_i, scheduler):
        """Fixed-dimension REPLACEMENT proposal on the picked ALIVE sources.

        Search heuristic (USER ruling, 2026-08-01; detailed balance waived
        for search): the dimension never changes and ``inds`` is untouched.
        Each picked alive leaf gets a fresh draw from the RJ proposal
        container (intrinsics from the fstat/rj container; dead slots are
        never drawn for), scored by
        :meth:`SubBandBuffer.get_replace_ll` -- a self-contained
        expose(old) -> score(both, phase-maximized) -> restore(bit-exact)
        wrapper -- and accepted on

            ln alpha = beta * [add(new) - add(old)]
                       + ln p_prior(new) - ln p_prior(old)
                       + proposal factors (container logpdf bookkeeping,
                         same convention as birth/death),

        where add(old) is the PHASE-MAXIMIZED comparison value only: on
        reject the old source keeps its ORIGINAL parameters exactly, and
        on accept the standard path applies the swap (subtract old's
        template, add new's) with the maximizing phase written back into
        new's phi0. ``ll_change_log`` records the EXACT residual-ll change
        (phase-maxed add(new) minus ACTUAL-phase add(old)).

        With ``rj_fstat_dist_birth`` the new draw's distance/phi0/iota/psi
        are recentered on the F-stat 4-parameter maximum computed on the
        EXPOSED residual: the parity class is open here, so the reference
        walker's parent residual holds the raw data in this band -- the
        Jaranowski-Krolak inversion sees the old source's full power. The
        old side evaluates the reverse density at its own recentered
        center (death convention).

        Cross-band replacements follow the RJ-birth convention: a new draw
        outside this cell's frequency window (band edges widened by N/4 on
        RJ buffers -- the same window the in-model repeats respect) is
        forbidden with ``-inf``.
        """
        xp = self.xp
        alive = band_sorter.inds[picked["ids"]]
        if not bool(alive.any()):
            return
        sel = xp.where(alive)[0]
        ids = picked["ids"][sel]
        slots = picked["slot_index"][sel]
        N_vals = picked["N_vals"][sel]
        t_i = picked["temp_inds"][sel]
        w_i = picked["walker_inds"][sel]
        b_i = picked["band_inds"][sel]
        l_i = band_sorter.leaf_inds[ids]
        n_prop = len(ids)

        params_old = band_sorter.coords[ids].copy()
        params_old[:] = self.periodic.wrap(
            {self.branch_name: params_old[:, None, :]}, xp=xp
        )[self.branch_name][:, 0]

        # Fresh replacement draw (NaN-repair loop mirrors the BandSorter
        # birth pre-draw). ``rj_prop`` is the resolved per-branch container.
        cont = band_sorter.rj_prop
        params_new = xp.full_like(params_old, np.nan)
        fix = xp.full(n_prop, True)
        while bool(xp.any(fix)):
            params_new[fix] = xp.asarray(cont.rvs(size=int(fix.sum().item())))
            fix = xp.any(xp.isnan(params_new), axis=-1)
        params_new[:] = self.periodic.wrap(
            {self.branch_name: params_new[:, None, :]}, xp=xp
        )[self.branch_name][:, 0]

        prev_logp = cp.asarray(self.gpu_priors[self.branch_name].logpdf(params_old))
        curr_logp = cp.asarray(self.gpu_priors[self.branch_name].logpdf(params_new))

        # Cross-band gate (RJ-birth convention; see docstring).
        f_hz = params_new[:, 1] / 1e3
        out_of_band = (
            (f_hz < buffer_obj.frequency_lims[0][slots])
            | (f_hz > buffer_obj.frequency_lims[1][slots])
        )
        curr_logp[out_of_band] = -np.inf

        # Proposal factors, existing-machinery convention: death side for
        # the replaced source is the sorter's precomputed +logpdf
        # (band_sorter.factors); birth side for the fresh draw is -logpdf.
        factors = band_sorter.factors[ids] - cp.asarray(cont.logpdf(params_new))

        keep = ~cp.isinf(curr_logp)
        delta_old = cp.full(n_prop, -1e300)
        delta_new = cp.full(n_prop, -1e300)
        delta_old_actual = cp.full(n_prop, -1e300)
        h_h_new = cp.zeros(n_prop)

        if self.rj_fstat_dist_birth and bool(keep.any()):
            # F-stat recentering-after-expose (preferred path): reuse the
            # birth machinery's helpers against the OPEN parent residual
            # (raw data in this parity's bands -> the old source's power is
            # exposed to the F-stat).
            walker_ref = getattr(self, "_fstat_walker_ref", 0)
            k_idx = xp.arange(n_prop)[keep]
            _log_range = self._log_dist_range(band_sorter)
            A_max, phi0_max, iota_max, psi_max, F = self._fstat_dist_centers(
                model, params_new[k_idx], walker_ref)
            ln_center, sigma = self._dist_center_and_width(
                params_new[k_idx], A_max, F)
            z = xp.asarray(cp.random.randn(len(k_idx)))
            ln_draw = ln_center + sigma * z
            if _gb_use_distance(self):
                params_new[k_idx, 0] = xp.exp(ln_draw)
            else:
                params_new[k_idx, 0] = ln_draw  # slot 0 is lnA already
            params_new[k_idx, 4] = xp.cos(iota_max % np.pi)
            params_new[k_idx, 5] = psi_max % np.pi
            params_new[k_idx, 3] = phi0_max % (2 * np.pi)
            _bl = self._slot0_log_proposal(params_new[k_idx, 0], ln_center, sigma)
            # Reverse side: old's slot-0 density about its OWN recentered
            # center (death convention). The +/- log_range pair cancels but
            # is kept for symmetry with the birth/death bookkeeping.
            Ad, _pd, _id, _psd, Fd = self._fstat_dist_centers(
                model, params_old[k_idx], walker_ref)
            ln_center_d, sigma_d = self._dist_center_and_width(
                params_old[k_idx], Ad, Fd)
            _dl = self._slot0_log_proposal(
                params_old[k_idx, 0], ln_center_d, sigma_d)
            factors[k_idx] = factors[k_idx] + (-_bl - _log_range) + (_dl + _log_range)
            # Re-evaluate the global prior at the recentered draw (f0 and
            # the band gate are unchanged by the overwrite).
            curr_logp[k_idx] = cp.asarray(
                self.gpu_priors[self.branch_name].logpdf(params_new[k_idx]))
            keep = ~cp.isinf(curr_logp)

        if bool(keep.any()):
            k_idx = xp.arange(n_prop)[keep]
            # GB_REPLACE_DEBUG=1: assert the wrapper is pure -- the residual
            # rows are BIT-IDENTICAL before the call and after it (which is
            # exactly the rejected-replacement invariant: on reject nothing
            # else touches the buffer).
            _replace_debug = os.environ.get("GB_REPLACE_DEBUG", "0") == "1"
            if _replace_debug:
                _rt_rows = xp.unique(slots[k_idx].astype(xp.int64))
                _rt_snap = buffer_obj.band_buffer[_rt_rows].copy()
            d_old, d_new, phase_new, d_old_act = buffer_obj.get_replace_ll(
                params_old[k_idx], params_new[k_idx], slots[k_idx],
                slots[k_idx], N_vals[k_idx],
                phase_maximize=self.phase_maximize, leaf_inds=l_i[k_idx],
            )
            if _replace_debug:
                _rt_after = buffer_obj.band_buffer[_rt_rows]
                assert bool(xp.all(_rt_after == _rt_snap)), (
                    f"{self.name}: get_replace_ll did not restore the "
                    "residual bit-exactly (expose/score/restore leak)."
                )
                logger.info(
                    "%s [GB_REPLACE_DEBUG] expose/score/restore round-trip "
                    "bit-identical on %d cell rows.",
                    self.name, int(_rt_rows.shape[0]),
                )
            delta_old[k_idx] = d_old
            delta_new[k_idx] = d_new
            delta_old_actual[k_idx] = d_old_act
            h_h_new[k_idx] = buffer_obj.replace_h_h_new
            # GB_DEBUG: keep the EXACT rows that were scored, before the
            # phi0 write-back mutates them, so the verifier can separate
            # "the write-back changed the answer" from "the scored value is
            # not reproducible at all".
            if self.debug:
                self._dbg_params_new_prewb = params_new.copy()
            if self.phase_maximize and phase_new is not None:
                # Maximizing phase into NEW's phi0 (the accepted parameters
                # carry it; a rejected old source is never re-phased).
                params_new[k_idx, self._phi0_col] = (
                    params_new[k_idx, self._phi0_col] - phase_new
                )

        # Independently re-verify the scored deltas through get_add_ll, the
        # same way _run_rj_step does via _debug_verify_rj_step. Replace was
        # the ONLY move without this hook -- _debug_verify_replace_swap
        # checks the RESIDUAL identity but never the likelihood, which is
        # why a ledger defect here stays invisible until it shows up as
        # propose-level drift.
        self._debug_verify_replace_step(
            buffer_obj, params_old, params_new, slots, N_vals, l_i,
            delta_old_actual, delta_new, keep,
        )

        delta_ll = cp.full(n_prop, -1e300)
        ok = keep & (delta_new > -1e299) & (delta_old > -1e299)
        delta_ll[ok] = delta_new[ok] - delta_old[ok]

        # SNR rejection-sampling clamp on the NEW side (add-side convention,
        # same ONE limit as births, applied to BOTH statistics -- optimal
        # sqrt(h_h) AND detected d_h/sqrt(h_h); user policy: effectively a
        # prior boundary): a sub-threshold replacement would silently
        # delete the source without death bookkeeping. d_h_new is recovered
        # from the add-convention delta (delta = d_h - 0.5*h_h on the
        # exposed residual). Skipped under the debug force-accept knob
        # (smoke-only residual-identity exercise).
        _force_accept = os.environ.get("GB_REPLACE_FORCE_ACCEPT", "0") == "1"
        if not _force_accept:
            opt_snr_new = xp.sqrt(xp.maximum(h_h_new, 0.0))
            _lim = buffer_obj.opt_snr_rej_samp_limit
            _bad_new = opt_snr_new < _lim
            if getattr(buffer_obj, "snr_rej_detected", False):
                det_snr_new = ((delta_new + 0.5 * h_h_new)
                               / xp.maximum(opt_snr_new, 1e-300))
                _bad_new = _bad_new | (det_snr_new < _lim)
            delta_ll[ok & _bad_new] = -1e300

        beta = band_temps[b_i, t_i]
        lnpdiff = beta * delta_ll + (curr_logp - prev_logp) + factors
        accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))

        bad_mask = (delta_ll <= -1e299) | (curr_logp <= -1e229)
        bad_accepts = accept & bad_mask
        if bool(xp.any(bad_accepts)):
            if bool(xp.any(beta[bad_accepts] != 0.0)) and not (
                "fstat" in self.name or "refit" in self.name
                or "replace" in self.name
            ):
                logger.warning(
                    f"{self.name}: accepted an out-of-prior REPLACE "
                    "coordinate at beta > 0."
                )
            accept[bad_accepts] = False
        if _force_accept:
            # Debug knob (GB_REPLACE_FORCE_ACCEPT=1, smoke only): accept
            # every finite replacement so the accepted-swap residual
            # identity can be asserted deterministically. NEVER for real
            # sampling.
            accept = ~bad_mask & (prev_logp > -1e229)

        prop_counts[0][t_i, w_i, b_i] += 1

        if bool(accept.any()):
            acc_ids = ids[accept]
            wrapped_new = self.periodic.wrap(
                {self.branch_name: params_new[accept][:, None, :]}, xp=xp
            )[self.branch_name][:, 0]

            _replace_debug = os.environ.get("GB_REPLACE_DEBUG", "0") == "1"
            _dbg_slot = _dbg_before = None
            if _replace_debug:
                _dbg_slot = int(slots[accept][0])
                _dbg_before = buffer_obj.band_buffer[_dbg_slot].copy()

            # STANDARD accept path applies the swap: subtract old's template
            # (= add it to the residual), add new's (= subtract it).
            buffer_obj.remove_sources_from_band_buffer(
                params_old[accept], slots[accept], N_vals[accept],
                leaf_inds=l_i[accept],
            )
            buffer_obj.add_sources_to_band_buffer(
                wrapped_new, slots[accept], N_vals[accept],
                leaf_inds=l_i[accept],
            )
            band_sorter.coords[acc_ids] = wrapped_new
            # inds untouched: the dimension never changes.

            if _replace_debug:
                self._debug_verify_replace_swap(
                    buffer_obj, _dbg_slot, _dbg_before,
                    params_old[accept][:1], wrapped_new[:1],
                    slots[accept][:1], N_vals[accept][:1],
                    None if l_i is None else l_i[accept][:1],
                )

            # Exact tracked ll change: phase-maxed add(new) minus the old
            # source's ACTUAL-phase add-delta (what the swap really removed).
            tracked = delta_new - delta_old_actual
            ll_change_log[t_i[accept], w_i[accept], b_i[accept]] += tracked[accept]
            acc_counts[0][t_i[accept], w_i[accept], b_i[accept]] += 1

    def _debug_verify_replace_step(self, buffer_obj, params_old, params_new,
                                   slots, N_vals, leaf_inds,
                                   delta_old_actual, delta_new, keep) -> None:
        """At the REPLACE scoring site: re-verify both scored deltas through
        ``get_add_ll`` on the exposed residual, the same way
        :meth:`_debug_verify_rj_step` re-verifies births and deaths.

        Walks the swap one stage at a time and reports the likelihood at
        each, so a mismatch localises itself:

        1. ``ll_expose`` -- add-delta of the OLD source once it is exposed
           (``r' = r + h_old``). Must equal ``delta_old_actual``. A mismatch
           means the old-side bookkeeping, not the swap.
        2. ``ll_new`` -- add-delta of the FINAL new parameters (after the
           phi0 write-back) against that same ``r'``. Must equal
           ``delta_new``, which was scored PHASE-MAXIMISED on the
           pre-write-back parameters. **A mismatch here means the phi0
           write-back does not reproduce the maximising template** -- the
           applied source differs from the scored one, so every accept
           mis-reports its ll change by the difference.

        Uses only ``get_add_ll`` and the buffer's own expose/restore path --
        no likelihood is reimplemented. The residual is restored from a
        snapshot, so this is a pure observation.
        """
        if not self.debug:
            return
        try:
            xp = self.xp
            k = _to_numpy(keep).astype(bool)
            if not k.any():
                return
            idx = xp.asarray(np.where(k)[0])
            di = slots[idx]
            li = None if leaf_inds is None else leaf_inds[idx]
            rows = xp.unique(xp.asarray(di).astype(xp.int64))
            snapshot = buffer_obj.band_buffer[rows].copy()
            try:
                # Stage 1: expose the old source -> r' = r + h_old.
                buffer_obj.remove_sources_from_band_buffer(
                    params_old[idx], di, N_vals[idx], leaf_inds=li)
                d_old_chk = buffer_obj.get_add_ll(
                    params_old[idx], di, di, N_vals[idx], leaf_inds=li)
                # Stage 2: the FINAL new parameters against the same r'.
                d_new_chk = buffer_obj.get_add_ll(
                    params_new[idx], di, di, N_vals[idx], leaf_inds=li)
                # Stage 2b: re-MAXIMISE on the final parameters. Phase
                # maximisation is invariant to phi0, so ``d_new_max`` must
                # reproduce ``delta_new`` exactly; the residual rotation
                # ``phase_angle`` must be ~0 if the write-back landed the
                # template on its maximum. This separates the two causes:
                #   |resid angle| ~ 0 but d_new_chk != delta_new  -> the
                #       maximised value itself is not attainable at ANY phi0
                #       (an amplitude/basis convention issue, not phi0).
                #   |resid angle| >> 0 -> the write-back is wrong.
                d_new_max = buffer_obj.get_add_ll(
                    params_new[idx], di, di, N_vals[idx], leaf_inds=li,
                    phase_maximize=True)
                resid_ang = getattr(buffer_obj, "phase_angle", None)
                resid_ang = None if resid_ang is None else resid_ang.copy()
                # Stage 2c: BATCHING control. The scored value came from a
                # 2n-row call ([old; new] sharing the same slot rows); the
                # re-maximised value above came from an n-row call. Score the
                # new half two more ways to separate "the batch layout" from
                # "the row content":
                #   cat_on:  concat([old, new])  -> new half at rows [n:]
                #   cat_dup: concat([new, new])  -> same 2n layout, no old
                # If cat_on's new half != solo but cat_dup's halves == solo,
                # the OLD rows are corrupting the NEW rows in a shared batch.
                _nn = int(idx.shape[0])
                _di2 = xp.concatenate([di, di])
                _nv2 = xp.concatenate([N_vals[idx], N_vals[idx]])
                _li2 = None if li is None else xp.concatenate([li, li])
                d_cat_on = buffer_obj.get_add_ll(
                    xp.concatenate([params_old[idx], params_new[idx]], axis=0),
                    _di2, _di2, _nv2, leaf_inds=_li2, phase_maximize=True)
                d_cat_dup = buffer_obj.get_add_ll(
                    xp.concatenate([params_new[idx], params_new[idx]], axis=0),
                    _di2, _di2, _nv2, leaf_inds=_li2, phase_maximize=True)
                # Stage 2d: the EXACT pre-write-back rows that were scored.
                # Phase maximisation is invariant to phi0, so this must equal
                # both ``delta_new`` and the post-write-back maximum. If it
                # equals delta_new but the post-write-back value differs, the
                # maximum is NOT phi0-invariant -- the quadrature partner is
                # not h(phi0 + pi/2) in the sampled basis.
                _pre = getattr(self, "_dbg_params_new_prewb", None)
                d_pre_max = None if _pre is None else buffer_obj.get_add_ll(
                    _pre[idx], di, di, N_vals[idx], leaf_inds=li,
                    phase_maximize=True)
                d_pre_act = None if _pre is None else buffer_obj.get_add_ll(
                    _pre[idx], di, di, N_vals[idx], leaf_inds=li)
                # Stage 2e: is the scored maximum ATTAINABLE at all? Scan the
                # actual-phase add-delta over a grid of phi0 and take the best.
                # The two-quadrature |D| is an analytic maximum that assumes
                # <r|h(phi0)> is exactly sinusoidal in phi0; the narrow m-band
                # truncation breaks that, so the true attainable maximum can
                # sit BELOW the analytic one. Uses only get_add_ll.
                _ngrid = 24
                _scan = None
                if _pre is not None:
                    _base = _pre[idx].copy()
                    _best = None
                    for _g in range(_ngrid):
                        _t = _base.copy()
                        _t[:, self._phi0_col] = (
                            _base[:, self._phi0_col] + 2 * np.pi * _g / _ngrid)
                        _v = buffer_obj.get_add_ll(
                            _t, di, di, N_vals[idx], leaf_inds=li)
                        _best = _v if _best is None else xp.maximum(_best, _v)
                    _scan = _best
            finally:
                buffer_obj.band_buffer[rows] = snapshot

            def _rel(a, b):
                fin = xp.isfinite(a) & xp.isfinite(b) & (b > -1e290)
                if not bool(fin.any()):
                    return float("nan")
                return float(xp.max(xp.abs(a[fin] - b[fin])
                                    / xp.maximum(xp.abs(b[fin]), 1.0)))

            r_old = _rel(d_old_chk, delta_old_actual[idx])
            r_new = _rel(d_new_chk, delta_new[idx])
            logger.info(
                "[GB_DEBUG %s] REPLACE old-side delta vs get_add_ll on the "
                "exposed residual: max rel %.3e", self.name, r_old,
            )
            logger.info(
                "[GB_DEBUG %s] REPLACE new-side delta (final phi0) vs "
                "get_add_ll: max rel %.3e  <- large here means the phi0 "
                "write-back does not reproduce the phase-maximised template",
                self.name, r_new,
            )
            # Localise the new-side mismatch: is the maximum reproducible at
            # all, and did the write-back land on it?
            r_max = _rel(d_new_max, delta_new[idx])
            if resid_ang is not None:
                _wr = (xp.abs((resid_ang + np.pi) % (2 * np.pi) - np.pi))
                _fin = xp.isfinite(_wr)
                ang = float(xp.max(_wr[_fin])) if bool(_fin.any()) else float("nan")
            else:
                ang = float("nan")
            if _scan is not None:
                # SIGNED: positive => the analytic maximum sits ABOVE anything
                # actually attainable, i.e. every phase-maximised score is
                # inflated and the ledger drifts up on every accept.
                _ex = delta_new[idx] - _scan
                _fs = xp.isfinite(_ex) & (delta_new[idx] > -1e290)
                logger.info(
                    "[GB_DEBUG %s] REPLACE attainability: scored max MINUS "
                    "best over a %d-point phi0 scan -> max %+.3e, median "
                    "%+.3e (positive => the two-quadrature maximum is NOT "
                    "attainable at any phi0; the ledger is inflated)",
                    self.name, _ngrid,
                    float(xp.max(_ex[_fs])) if bool(_fs.any()) else float("nan"),
                    float(xp.median(_ex[_fs])) if bool(_fs.any()) else float("nan"),
                )
            if d_pre_max is not None:
                logger.info(
                    "[GB_DEBUG %s] REPLACE phi0-invariance: PRE-write-back max "
                    "vs scored %.3e | PRE max vs POST max %.3e | PRE actual-"
                    "phase vs POST max %.3e  (col2 large => the maximum is not "
                    "phi0-invariant, i.e. h(phi0+pi/2) is not the quadrature "
                    "partner in the SAMPLED basis)",
                    self.name,
                    _rel(d_pre_max, delta_new[idx]),
                    _rel(d_pre_max, d_new_max),
                    _rel(d_pre_act, d_new_max),
                )
            logger.info(
                "[GB_DEBUG %s] REPLACE batching control: cat[old;new] new-half "
                "vs solo %.3e | cat[new;new] first-half vs solo %.3e | "
                "cat[new;new] halves self-consistent %.3e  (first large + rest "
                "~0 => old rows corrupt new rows in a shared batch)",
                self.name,
                _rel(d_cat_on[_nn:], d_new_max),
                _rel(d_cat_dup[:_nn], d_new_max),
                _rel(d_cat_dup[_nn:], d_cat_dup[:_nn]),
            )
            logger.info(
                "[GB_DEBUG %s] REPLACE new-side split: remaximised vs scored "
                "max rel %.3e | residual phase angle at final phi0 max |ang| "
                "%.3e rad  (angle~0 + rel>0 => the scored maximum is not "
                "attainable at any phi0; angle>>0 => write-back is wrong)",
                self.name, r_max, ang,
            )
            # Absolute ll scale of the swap, for comparison against the
            # propose-level drift this move reports.
            tracked = (delta_new[idx] - delta_old_actual[idx])
            direct = (d_new_chk - d_old_chk)
            fin = xp.isfinite(tracked) & xp.isfinite(direct)
            if bool(fin.any()):
                logger.info(
                    "[GB_DEBUG %s] REPLACE tracked vs direct swap dll: max "
                    "ABS dev %.3e (tracked |max| %.3e)", self.name,
                    float(xp.max(xp.abs(tracked[fin] - direct[fin]))),
                    float(xp.max(xp.abs(tracked[fin]))),
                )
        except Exception as e:  # debug-only: never break the sampler
            logger.warning(
                "[GB_DEBUG %s] verify_replace_step skipped: %r", self.name, e)

    def _debug_verify_replace_swap(self, buffer_obj, slot, r_before,
                                   p_old, p_new, slot_arr, n_arr, l_arr):
        """GB_REPLACE_DEBUG=1: assert an accepted replacement changed the
        residual by exactly ``h_old - h_new`` (direct fill_template
        comparison on one cell). The templates are materialized by filling
        each source into the zeroed slot row; the row is restored to the
        post-swap state afterwards."""
        xp = self.xp
        r_after = buffer_obj.band_buffer[slot].copy()
        try:
            buffer_obj.band_buffer[slot] = xp.zeros_like(r_after)
            # factor=+1 through the standard fill path -> +h_old in the row.
            buffer_obj.remove_sources_from_band_buffer(
                p_old, slot_arr, n_arr, leaf_inds=l_arr)
            h_old = buffer_obj.band_buffer[slot].copy()
            buffer_obj.band_buffer[slot] = xp.zeros_like(r_after)
            buffer_obj.remove_sources_from_band_buffer(
                p_new, slot_arr, n_arr, leaf_inds=l_arr)
            h_new = buffer_obj.band_buffer[slot].copy()
        finally:
            buffer_obj.band_buffer[slot] = r_after
        expected = (r_before + h_old) - h_new
        diff = float(xp.abs(r_after - expected).max())
        scale = float(xp.abs(expected).max()) or 1.0
        assert diff <= 1e-12 * scale, (
            f"{self.name}: accepted replacement residual identity violated: "
            f"max |r_after - (r_before + h_old - h_new)| = {diff:.3e} "
            f"(scale {scale:.3e})"
        )
        logger.info(
            "%s [GB_REPLACE_DEBUG] accepted-swap residual identity OK: "
            "max abs dev = %.3e (scale %.3e, slot %d)",
            self.name, diff, scale, slot,
        )

    def _ensure_proposal_tables(self, model, band_sorter):
        """Rebuild the cold-chain proposal tables for THIS proposal.

        Called from the first in-model block of the proposal -- i.e. between
        this move's RJ step and its in-model sequence, so both tables describe
        the post-RJ cold chain -- and a no-op on every later block of the same
        proposal.

        Both products come from the cold chain only (``inds & temp_inds == 0``,
        all walkers): the group-stretch friend table (frequency-sorted
        coordinates) and the info-matrix Cholesky table (frequency-sorted
        proposal factors). Every source at every temperature then indexes into
        them, so the factor cost is one cold chain rather than one per
        (temp, walker) block -- roughly a factor ``ntemps`` fewer information
        matrices per proposal, while each RJ proposal still gets factors built
        at its own post-RJ parameters.
        """
        if self._tables_indexed:
            return
        self._tables_indexed = True

        if self.share_proposal_tables:
            # TODO: LOOK AT THIS MORE CLOSELY BEFORE ENABLING. Reusing one
            # table across proposals (e.g. once per larger iteration, shared
            # by the fstat-birth -> replace -> prior-removal cycle) removes
            # most of the remaining cost, and _SharedProposalTables /
            # _shared_proposal_tables below implement the plumbing for it. It
            # is OFF because the staleness is not understood: RJ births and
            # deaths change the cold chain between the moves of a cycle, so a
            # later move would index into a table describing a source
            # population that no longer exists, and the factors themselves
            # drift as the fit evolves. Needs an acceptance-rate comparison
            # against per-proposal rebuilds before it can be trusted.
            raise NotImplementedError(
                "share_proposal_tables=True is not implemented: sharing the "
                "cold-chain friend / info-matrix tables ACROSS proposals is "
                "unvalidated (the cold chain changes between the RJ moves of "
                "a cycle). Leave it False; see the TODO in "
                "_ensure_proposal_tables."
            )

        tm = getattr(self, "_prop_timer", None)
        with _tspan(tm, "proposal_tables"):
            if self._build_friend_table and self.stretch_probability > 0.0:
                band_sorter.index_friends(
                    band_sorter.build_friend_table(self.nfriends), self.nfriends)
            if self.use_info_mat_proposal:
                # GB_INFOMAT_PER_BLOCK=1 RETIRES the borrow: skip the
                # cold-chain table entirely so ``_proposal_cholesky`` takes
                # its existing direct branch and every block computes EXACT
                # matrices for its OWN sources. Still built ONCE per block,
                # before the repeats and held constant during them. Only the
                # info-matrix half is skipped -- the group-stretch FRIEND
                # table above is a separate product and is still required.
                # Affordable only with the sig-het path (SIGHET_INFOMAT=1):
                # ~2.4 ms/source instead of 46.44, which makes per-block
                # exact CHEAPER than the borrowed table (54 s vs 115 s per
                # proposal) as well as correct.
                if os.environ.get("GB_INFOMAT_PER_BLOCK", "0") == "1":
                    self._infomat_freqs_sorted = None
                    self._infomat_chol_sorted = None
                    logger.info(
                        "%s: per-block EXACT info matrices (borrow retired)",
                        self.name)
                else:
                    self._refresh_infomat_table(model, band_sorter)
                    band_sorter.build_infomat_index(
                        self._infomat_freqs_sorted, self._infomat_chol_sorted)

    def _refresh_infomat_table(self, model, band_sorter):
        """Rebuild the shared cold-chain info-matrix Cholesky table.

        The group-stretch friend table's counterpart for proposal COVARIANCES.
        The cold-chain sources (``inds & temp_inds == 0``, all walkers) are
        sorted in frequency and their Cholesky factors computed in ONE batched
        call; the table then serves every source at every temperature for this
        proposal's in-model sequence, each one taking the nearest entry in
        frequency (see :meth:`BandSorter.build_infomat_index`).

        Two properties make this sound enough to be worth the approximation:
        the information matrix depends only on the parameters and the noise --
        never on the residual -- so a table entry does not go stale as the fit
        subtracts sources; and the factor is only a proposal SHAPE, which M-H
        corrects. It is the same class of approximation as drawing a
        group-stretch partner from a frequency window.

        # TODO: REVIEW THIS PROCESS -- the information matrices change as the
        # fit evolves, and a table refreshed only every N proposals hands
        # sources a covariance built at parameters (and, for the hot chains, at
        # a temperature and a walker's noise realisation) that are not their
        # own. Worth checking: the acceptance-rate cost of the refresh cadence
        # vs computing per block; whether nearest-in-frequency is the right
        # partner rule once sources differ substantially in SNR (the factor
        # scales with amplitude, so a loud neighbour hands a quiet source a
        # badly-sized jump); and whether the cold-chain table should be
        # temperature-corrected before the hot rungs borrow from it.
        """
        cold = band_sorter.inds & (band_sorter.temp_inds == 0)
        n_cold = int(cold.sum())
        if n_cold == 0:
            self._infomat_freqs_sorted = None
            self._infomat_chol_sorted = None
            return False

        cold_ids = self.xp.where(cold)[0]
        order = self.xp.argsort(band_sorter.coords[cold_ids, 1])
        cold_ids = cold_ids[order]
        # Chunked: the per-block call this replaces passed 1-2 sources, while
        # the whole cold chain is (nwalkers x live leaves) and each source
        # costs ~17 waveform evaluations, so one unbounded call would spike
        # peak device memory at production leaf counts.
        step = max(int(self.infomat_table_batch), 1)
        parts = [
            self._compute_proposal_cholesky(model, band_sorter, cold_ids[i:i + step])
            for i in range(0, n_cold, step)
        ]
        self._infomat_chol_sorted = (
            parts[0] if len(parts) == 1 else self.xp.concatenate(parts, axis=0)
        )
        self._infomat_freqs_sorted = band_sorter.coords[cold_ids, 1].copy()
        logger.info(
            "%s: info-matrix table rebuilt from %d cold-chain sources "
            "(iteration %d)", self.name, n_cold, self.time,
        )
        return True

    def _proposal_cholesky(self, model, band_sorter, ids, slots=None,
                           buffer_obj=None):
        """Proposal Cholesky factors for ``ids``: table lookup, else direct.

        Falls back to the direct per-block computation whenever the table is
        unavailable -- a cold chain with no live sources, or a caller that
        reaches the in-model step without ``_ensure_proposal_tables``.

        ``slots`` are the per-source buffer slots, aligned row-for-row with
        ``ids`` (both are ``picked[...][alive]``). They are what the sig-het
        fast info-matrix route needs; ``None`` (the shared-table path, which
        spans the whole cold chain and has no single block's slots) keeps the
        validated chunked route. ``buffer_obj`` (the bound SubBandBuffer) is
        forwarded as the router's ``slot_holder`` so multi-shard runs
        partition ``slots`` by the BUFFER's slot shards -- the shard layout
        ``setup_in_model`` actually stashed references under.
        """
        if getattr(band_sorter, "infomat_take_inds", None) is None:
            return self._compute_proposal_cholesky(
                model, band_sorter, ids, slots=slots, buffer_obj=buffer_obj)
        # The direct path sets these as a side effect; the table path must
        # set them too (in_model_proposal maps the drawn jump back with them).
        s = self.xp.ones(band_sorter.coords.shape[1])
        if self._fdot_col is not None:
            s[self._fdot_col] = self._fdot_scale
        self._proposal_param_scales = s
        return band_sorter.draw_infomat(ids)

    def _compute_proposal_cholesky(self, model, band_sorter, ids, slots=None,
                                   buffer_obj=None):
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
        # Sub-spans: ``inmodel_cholesky`` was 84% of the overnight_v5
        # iteration and lumps three very different costs together (the
        # information-matrix kernel, the numerical Jacobian, and the
        # eigendecomposition). Split them so the next run attributes it.
        # The sig-het fast route (SIGHET_INFOMAT) scores through get_ll_wdm
        # against the in-model reference, and while that reference is live
        # ``data_index`` means the per-source BUFFER SLOT -- not the walker.
        # Raw comps take no such argument, so it goes only to the sig-het
        # wrapper (identified by its ``.chunked`` delegate, the same idiom
        # _fstat_NM uses). Without slots the wrapper falls back to chunked,
        # which is what the shared-table caller wants.
        _di = {}
        if slots is not None and hasattr(_info_comp, "chunked"):
            _di["data_index"] = xp.asarray(slots, dtype=xp.int32)
            if buffer_obj is not None:
                # Slot-space routing (sig-het fast leg): the reference
                # stash lives on the BUFFER's per-device comp replicas,
                # keyed intra-shard -- the router needs the buffer to
                # partition slots coherently on multi-shard runs.
                # Ignored by the router on single-shard buffers.
                _di["slot_holder"] = buffer_obj

        _tm = getattr(self, "_prop_timer", None)
        _t_im0 = time.perf_counter()
        with _tspan(_tm, "infomat_kernel"):
            info_phys = _RoutedBandEngine.route_information_matrix(
                _info_comp, model.analysis_container_arr, params_phys,
                inds=_test_inds, noise_index=walker_inds, **_di,
            )
        # ROUTE INDICATOR. The fall-through to the chunked delegate is
        # SILENT -- unlike the F-stat route, nothing warns -- and it costs
        # ~46 ms/source against ~2.4 ms on the sig-het route (faster still
        # under v5). That invisibility is what burned the overnight_v5 run:
        # it asked for v5, ``inmodel_cholesky`` was 84% of the iteration,
        # and no log line could show whether sig-het was ever reached. The
        # per-source cost is the only available detector, so make it one.
        self._infomat_route_check(
            time.perf_counter() - _t_im0, int(params_phys.shape[0]),
            fast_wired=bool(_di), comp=_info_comp)

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
        # Numerical diagonal Jacobian d(phys[test_inds[i]]) / d(y_i) through
        # the transform container -- generic in the container's transforms.
        #
        # NOTE(infomat-jacobian-batching): this runs 2*ndim separate
        # ``both_transforms`` calls (18 on the 9-column basis) on the full
        # (n_src, ndim) block, which looks like an obvious batching target
        # -- stack every perturbed copy into one (2*ndim*n_src, ndim) call.
        # An attempt at that did NOT reproduce this loop (two columns off by
        # ~4e-2, far too large for roundoff, cause not isolated), so it is
        # deliberately left alone: this feeds the proposal covariance and a
        # silently-wrong Jacobian would bias every in-model jump. The
        # ``infomat_jacobian`` span measures whether it is worth revisiting.
        with _tspan(_tm, "infomat_jacobian"):
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

        # Opt-in only (perf, 2026-08-15): this sat INSIDE the per-block
        # info-matrix path, so every in-model block paid a full CuPy pool
        # release -- cudaFree/cudaMalloc churn for every allocation that
        # follows (same rationale as GB_MEMPOOL_FREE_EACH_ROUND). The
        # per-unit / per-proposal frees remain unconditional.
        if os.environ.get("GB_INFOMAT_MEMPOOL_FREE", "0") == "1":
            self.mempool.free_all_blocks()
        # Robust inverse-information-matrix factor: near-zero-SNR (prior-drawn) sources
        # give (numerically) singular information matrices. Eigendecompose and clamp the
        # spectrum to a relative floor; B = V diag(lambda^-1/2) satisfies
        # B B^T = inv(info) and is all the Gaussian proposal needs (the
        # proposal shape only -- M-H corrects).
        with _tspan(_tm, "infomat_eigh"):
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

        self._last_im_kind = "stretch" if use_stretch else "infomat"
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
                              picked, ll_change_log, prop_counts, acc_counts,
                              num_repeats=None):
        """``num_repeats`` in-model rounds on the picked live sources.

        The picked source is first taken OUT of its cell residual, so every
        repeat scores through a plain ``get_add_ll`` against the
        source-free residual (the buffer is not touched between repeats --
        only the tracked coordinates and counters move). After the repeats,
        the final coordinates are written back into the residual and into
        the BandSorter.

        ``num_repeats`` (None = ``self.num_repeat_proposals``) is the FIXED
        repeat budget for this block -- the RJ round machinery passes the
        per-provenance-class budgets here (newborn vs mature, user ruling
        2026-08-15) while pure in-model moves keep the plain knob.

        De-synced accept chain (perf, 2026-08-15): on CuPy every
        ``bool(...)`` / ``int(...)`` on a device scalar and every
        boolean-mask getitem is a forced device sync, and this loop paid
        ~14-16 of them PER REPEAT. The loop now keeps the accept masks and
        per-kind counters ON DEVICE across repeats -- ONE data-dependent
        sync per repeat remains (the ``xp.where(keep)`` compression that
        sizes the scoring kernel's row set) and the counter host-pulls
        aggregate once per block (``imr_accept_flush``). Accept/reject
        DECISIONS are bit-identical to the straight-line form: the RNG
        draw count/order/shape and the MH-ratio arithmetic are untouched;
        gated host branches became unconditional masked device ops with
        identical results.
        """
        xp = self.xp
        n_rep = (
            int(num_repeats) if num_repeats is not None
            else int(self.num_repeat_proposals)
        )
        # Cold-chain proposal tables (friends + info-matrix Cholesky) are
        # built HERE, on the first in-model block of the proposal: that point
        # is after this iteration's first RJ step, so the tables describe the
        # post-RJ cold chain, and they are shared with the other GB moves of
        # the cycle. A no-op on every later block.
        self._ensure_proposal_tables(model, band_sorter)
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

        # Per-source likelihood setup for the repeat block (same stage as
        # the proposal cholesky / friend table). Chunked-het / FD engines
        # no-op; a sig-het computation builds its heterodyne reference
        # against the source-free residual HERE and holds it CONSTANT for
        # the whole repeat block, so ll_ref below and every repeat's
        # get_add_ll score through the same likelihood.
        #
        # ORDER (2026-08-09): this MUST precede the proposal Cholesky below.
        # ``GBSignalHetComputations.information_matrix`` only takes its fast
        # ``SIGHET_INFOMAT`` route -- second differences of the likelihood
        # through ``get_ll_wdm``, i.e. at sig-het (v5) speed against the
        # reference this call builds -- while ``_in_model`` is live. Built
        # the other way round it silently fell through to the chunked
        # delegate at ~46 ms/source instead of ~2.4 ms, which is why
        # ``inmodel_cholesky`` was 84% of the overnight_v5 iteration and why
        # that run measured NO v5 gain: v5 accelerates sig-het scoring, and
        # the dominant cost never reached sig-het. The reference is built
        # against the SOURCE-FREE residual, which is exactly the right one
        # for that source's own curvature.
        with _tspan(tm, "inmodel_sighet_setup"):
            sighet_active = bool(
                buffer_obj.setup_in_model_likelihood(curr, slots, N_vals, leaf_inds=l_i)
            )
        # Pure-stretch moves (use_info_mat_proposal=False, e.g. the fixed-
        # dimensional VGB move) never build the info-matrix Cholesky.
        with _tspan(tm, "inmodel_cholesky"):
            chol = (
                self._proposal_cholesky(model, band_sorter, ids, slots=slots,
                                        buffer_obj=buffer_obj)
                if self.use_info_mat_proposal
                else None
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
        # Tobs only exists on time-frequency bases (WDM/STFT) — and is only
        # consumed by the trust-region gate math, which is active exactly
        # when ``anchor_phys`` is (dev regression: the unconditional read
        # broke every FD-domain GB flow).
        trust_Tobs = (
            float(self._basis_settings.Tobs) if anchor_phys is not None else 0.0
        )
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

        # Per-half repeat-INVARIANT gathers, hoisted out of the repeat loop
        # (they were re-gathered every repeat). Integer/slice indexing only
        # -- no data-dependent host syncs -- and each ``_n_cold`` is the
        # block's ONLY host pull for the proposed-cold counter (it was an
        # ``int((t_i[sl] == 0).sum())`` device pull per repeat).
        _half_pre = []
        for _sub in halves:
            _sl = slice(None) if _sub is None else _sub
            _t_s = t_i[_sl]
            _cold_s = _t_s == 0
            _half_pre.append((
                _sub, _sl,
                len(ids) if _sub is None else int(_sub.size),
                ids[_sl], slots[_sl], N_vals[_sl], l_i[_sl],
                _t_s, w_i[_sl], b_i[_sl], beta[_sl],
                n4[_sl], lo_bin[_sl], hi_bin[_sl],
                _cold_s, int(_cold_s.sum()),
            ))

        # Device-resident accept-chain state (flushed ONCE per block in
        # ``imr_accept_flush`` below): per-proposal-kind counters
        # [proposed, accepted(dev), cold-proposed, cold-accepted(dev)] and
        # the out-of-prior bad-accept census (warning deferred to block
        # end; it was a bool() sync PAIR per repeat).
        _kind_acc = {}
        _warn_dev = xp.zeros((), dtype=xp.int64)

        for move_i in range(n_rep):
          for (sub, sl, n_sub, ids_s, slots_s, N_s, l_s, t_s, w_s, b_s,
               beta_s, n4_s, lo_s, hi_s, cold_s, n_cold_s) in _half_pre:
            if sub is not None:
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
                    # NOTE(trust-gate hoisting): the anchor side of this
                    # gate is already hoisted to block scope (anchor_phys /
                    # trust_dlna above); the candidates change EVERY repeat
                    # so their transform is inherently per-repeat.
                    _pc = self.transform_fn.both_transforms(
                        new, xp=cp,
                        leaf_inds=l_s if self._per_leaf_fill else None,
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
                # THE one data-dependent host sync this repeat needs on
                # CuPy: compress ``keep`` ONCE and integer-gather through
                # it everywhere below. Each boolean-mask getitem it
                # replaces (5-6 per repeat) re-ran nonzero + a D2H size
                # pull; ``keep_idx`` yields the same rows in the same
                # (ascending) order, so every downstream value is
                # bit-identical.
                keep_idx = xp.where(keep)[0]
                keep_any = int(keep_idx.size) > 0
            new_ll = cp.full(n_sub, -1e300)
            # THE per-repeat scoring call: the sig-het fused in-kernel
            # likelihood when a reference is active, the chunked-het/FD
            # engine otherwise. This span is the headline number for the
            # in-model GB/GB speedup work.
            with _tspan(tm, "inmodel_get_add_ll"):
                if keep_any:
                    new_ll[keep_idx] = buffer_obj.get_add_ll(
                        new[keep_idx], slots_s[keep_idx], slots_s[keep_idx],
                        N_s[keep_idx],
                        phase_maximize=self.phase_maximize,
                        leaf_inds=l_s[keep_idx],
                    )
                    if self.phase_maximize and buffer_obj.phase_angle is not None:
                        new[keep_idx, self._phi0_col] = (
                            new[keep_idx, self._phi0_col]
                            - buffer_obj.phase_angle
                        )
                        new[keep_idx] = self.periodic.wrap(
                            {self.branch_name: new[keep_idx][:, None, :]},
                            xp=xp,
                        )[self.branch_name][:, 0]
            if tm is not None:
                tm.count("inmodel_repeat_calls")

            delta_ll = new_ll - ll_ref[sl]

            # SNR prior-boundary clamp on IN-MODEL updates (user policy
            # 2026-08-02: ONE limit, optimal sqrt(h_h) AND detected
            # d_h/sqrt(h_h), enforced on ALL GB moves as effective prior
            # support). Applies to the NEW point only, so a source already
            # below the limit can still move OUT of the violating region;
            # it can never move further in or laterally within it.
            # d_h_out/h_h_out are the per-repeat get_add_ll outputs for the
            # ``keep`` subset (the same arrays the sorter stash consumes).
            # ``keep_any`` is REQUIRED here, not just defensive: get_add_ll
            # above runs only under that same condition, so a repeat whose
            # candidates are all prior/trust-rejected leaves d_h_out/h_h_out
            # holding the PREVIOUS call's rows. Without this guard the clamp
            # would scatter a stale-length mask through ``keep_idx``.
            if getattr(buffer_obj, "d_h_out", None) is not None and keep_any:
                _hh_im = cp.asarray(buffer_obj.h_h_out).real
                _opt_im = cp.sqrt(cp.maximum(_hh_im, 0.0))
                _lim_im = buffer_obj.opt_snr_rej_samp_limit
                _viol_im = _opt_im < _lim_im
                if getattr(buffer_obj, "snr_rej_detected", False):
                    _dh_im = cp.asarray(buffer_obj.d_h_out).real
                    _det_im = _dh_im / cp.maximum(_opt_im, 1e-300)
                    _viol_im = _viol_im | (_det_im < _lim_im)
                # Unconditional masked write (was a bool(any) host gate +
                # a boolean-index scatter): the violating rows are exactly
                # ``keep_idx[_viol_im]``; non-violating rows are rewritten
                # with their own value, and the delta recompute repeats the
                # identical subtraction -- bit-identical either way.
                new_ll[keep_idx] = cp.where(_viol_im, -1e300, new_ll[keep_idx])
                delta_ll = new_ll - ll_ref[sl]

            # Device-resident MH bookkeeping (2026-08-15). This span used to
            # be the launch-overhead signal because every ``bool(...any())``
            # / ``int(...)`` in it was a device sync; the whole chain now
            # stays on device -- unconditional masked ops with results
            # identical to the gated branches they replace -- and the
            # counters flush once per block (``imr_accept_flush``).
            with _tspan(tm, "inmodel_accept"):
                lnpdiff = beta_s * delta_ll + (new_logp - curr_prior[sl]) + factors
                accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))

                # GB_JUMP_TRACE: the ONE site where the proposed coordinates,
                # the current ones, the accept mask and the per-row rung all
                # coexist. Device-only; no-op unless the knob is set.
                self._jump_trace_accum(
                    new, curr[sl], accept, t_i[sl], keep_idx,
                    getattr(buffer_obj, "h_h_out", None))

                bad_mask = (new_ll <= -1e299) | (new_logp <= -1e229)
                # ``accept[bad_accepts] = False`` == ``accept & ~bad_mask``;
                # the out-of-prior warning census accumulates on device and
                # logs ONCE at block end instead of per repeat.
                _warn_dev = _warn_dev + (
                    (accept & bad_mask) & (beta_s != 0.0)
                ).sum()
                accept = accept & ~bad_mask

                prop_counts[1][t_s, w_s, b_s] += 1
                # Per-proposal-type acceptance (stretch vs info-matrix):
                # the pooled counter cannot say WHICH proposal is timid.
                # Proposed tallies are host ints (shapes hoisted above);
                # accepted tallies stay 0-d device scalars until the flush.
                _kind = getattr(self, "_last_im_kind", None)
                if _kind is not None:
                    rec = _kind_acc.setdefault(_kind, [0, 0, 0, 0])
                    rec[0] += n_sub
                    rec[2] += n_cold_s
                    rec[1] = rec[1] + accept.sum()
                    rec[3] = rec[3] + (accept & cold_s).sum()

                # Unconditional masked accept application: ``cp.where``
                # copies the accepted values verbatim (rejected rows keep
                # their own), so the tracked state is bit-identical to the
                # boolean-scatter form it replaces.
                _tgt = slice(None) if sub is None else sub
                curr[_tgt] = cp.where(accept[:, None], new, curr[_tgt])
                ll_ref[_tgt] = cp.where(accept, new_ll, ll_ref[_tgt])
                curr_prior[_tgt] = cp.where(accept, new_logp, curr_prior[_tgt])
                # One pooled survivor per cell (serial-within-band), so the
                # fancy-index += is elementwise; rejected rows add an exact
                # 0.0 / False.
                ll_change_log[t_s, w_s, b_s] += cp.where(accept, delta_ll, 0.0)
                acc_counts[1][t_s, w_s, b_s] += accept
                if (
                    getattr(self, "_sorter_dh", None) is not None
                    and getattr(buffer_obj, "d_h_out", None) is not None
                    and keep_any
                ):
                    # d_h_out/h_h_out hold the per-repeat get_add_ll
                    # outputs for the ``keep_idx`` rows; scatter them to
                    # full width, then masked-write the accepted rows.
                    # ``accept`` implies keep (bad_mask filtering above),
                    # so the uninitialized non-keep lanes are never
                    # selected.
                    _dh_full = cp.empty(n_sub)
                    _hh_full = cp.empty(n_sub)
                    _dh_full[keep_idx] = cp.asarray(buffer_obj.d_h_out).real
                    _hh_full[keep_idx] = cp.asarray(buffer_obj.h_h_out).real
                    self._sorter_dh[ids_s] = cp.where(
                        accept, _dh_full, self._sorter_dh[ids_s]
                    )
                    self._sorter_hh[ids_s] = cp.where(
                        accept, _hh_full, self._sorter_hh[ids_s]
                    )

            # Guard at the CALL SITE (perf, 2026-08): the callee also checks
            # ``self.debug``, but its arguments — three ``asnumpy`` device
            # pulls — were evaluated unconditionally on EVERY in-model
            # repeat, each a D2H sync even with debug off.
            if self.debug:
                self._debug_verify_in_model(
                    buffer_obj, curr[sl], new, slots_s, N_s, delta_ll, keep,
                    (asnumpy(t_s), asnumpy(w_s), asnumpy(b_s)), move_i,
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
                and move_i + 1 < n_rep
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

        # ONE host pull per BLOCK for the accept-chain bookkeeping the loop
        # kept on device: fold the per-kind device tallies into the
        # per-propose ``_im_kind_counts`` dict and emit the deferred
        # out-of-prior warning (per-repeat before 2026-08-15; the block
        # total replaces up to ``n_rep`` identical lines).
        with _tspan(tm, "imr_accept_flush"):
            if _kind_acc:
                kc = getattr(self, "_im_kind_counts", None)
                if kc is None:
                    kc = self._im_kind_counts = {}
                for _k, _r in _kind_acc.items():
                    rec = kc.setdefault(_k, [0, 0, 0, 0])  # p, a, p0, a0
                    rec[0] += int(_r[0])
                    rec[1] += int(_r[1])
                    rec[2] += int(_r[2])
                    rec[3] += int(_r[3])
            _n_warn = int(_warn_dev)
            if _n_warn > 0:
                logger.warning(
                    f"{self.name}: accepted {_n_warn} out-of-prior in-model "
                    "coordinate(s) at beta > 0 in this repeat block."
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
                f"{n_rep} repeats): phase max="
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

    def _tempering_swap_grid(self, band_sorter, start, units=2):
        """Permuted (band, walker, temp) cell grid for one tempering unit.

        Interior bands only (the edge bands host no swaps), every
        temperature, and an independent random walker permutation per
        (band, temp) -- adjacent temperature columns of a grid row are the
        cells whose templates may exchange. Only the ``start``-unit
        interior bands (``arange(1, nb - 1)[start::units]``) are kept;
        ``units = 2`` (the default) is the legacy parity behavior,
        bit-identical. Swaps exchange templates BETWEEN TEMPERATURES of
        one band -- never across bands -- so a unit's concurrently-open
        bands only need the same orthogonality separation the proposal
        units use (stride guard in the ctor).

        Returns ``(band_index, temp_index, walkers_permuted, special_index,
        num_bands_unit)``; the first four are shaped
        ``(bands_this_unit, nwalkers, ntemps)``.
        """
        if self.num_bands == 1:
            num_bands_tempered = 1
            band_index_arr = cp.arange(1)
        else:
            num_bands_tempered = self.num_bands - 2
            band_index_arr = cp.arange(1, self.num_bands - 1)

        num_bands_unit = np.arange(num_bands_tempered)[start::units].shape[0]

        walkers_permuted = (
            cp.asarray(
                [
                    self._permute_walkers_for_swaps()
                    for _ in range(self.ntemps * num_bands_tempered)
                ]
            )
            .reshape(num_bands_tempered, self.ntemps, self.nwalkers)
            .transpose(0, 2, 1)[start::units]
        )
        temp_index = (
            cp.repeat(cp.arange(self.ntemps), num_bands_tempered * self.nwalkers)
            .reshape(self.ntemps, num_bands_tempered, self.nwalkers)
            .transpose(1, 2, 0)[start::units]
        )
        band_index = (
            cp.repeat(band_index_arr, self.ntemps * self.nwalkers)
            .reshape(num_bands_tempered, self.ntemps, self.nwalkers)
            .transpose(0, 2, 1)[start::units]
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

        # GB_TEMPER_AUDIT=1: reconcile the credited per-cold-walker ll
        # deltas (what lands in ll_change_log_temp[0] and, from there, in
        # state.log_like[0]) against the TRUE parent-residual likelihood at
        # every unit boundary, and print each ACCEPTED cold-pair swap with
        # its credited diff. Chasing the after-tempering incremental-ll
        # drift (sign-consistent ~ -<h|h>/2 per walker; reproduces on ONE
        # GPU, so it is accounting, not a device race). Successor to the
        # ll_before3/ll_after3 remnants below.
        _audit = os.environ.get("GB_TEMPER_AUDIT", "0") == "1"
        if _audit:
            _audit_true_prev = _to_numpy(model.analysis_container_arr.likelihood())
            _audit_cred_prev = np.zeros(self.nwalkers)

        # Band-unit stride (GB_BAND_UNIT_STRIDE / band_units ctor kwarg):
        # generalizes the historical hard-coded parity 2. Same stride as
        # the proposal units in run_proposal, so the concurrently-open
        # band separation guarantee -- (stride - 1) bands >= 1 WDM layer
        # (orthogonality; see check_band_support_separation) -- holds
        # here too. Swaps only exchange templates between temperatures
        # WITHIN a band, so every band still receives its swaps across
        # the ``units`` sequential passes. num_bands == 1 keeps the
        # legacy 2-pass loop verbatim (degenerate single-band case).
        units = self.band_units if self.num_bands > 1 else 2
        tmp_start = np.random.randint(units)
        for tmp in range(units):
            remainder = (tmp_start + tmp) % units
            start = remainder
            # Open exactly the band class the grid below selects:
            # interior bands arange(1, nb-1)[start::units] have
            # band % units == (start + 1) % units (tempering begins at
            # band 1). At units == 2 this is the legacy
            # ``bool_remainder = 1 if start == 0 else 0``.
            bool_remainder = _tempering_open_remainder(start, units)

            with _tspan(getattr(self, "_prop_timer", None), "temper_open_close"):
                self.remove_cold_chain_sources_from_residual(
                    model,
                    band_sorter,
                    extra_bool=(band_sorter.band_inds % units == bool_remainder),
                )

            (band_index, temp_index, walkers_permuted, special_index,
             num_bands_unit) = self._tempering_swap_grid(
                 band_sorter, start, units=units)

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

                    # Audit BEFORE current_lls is overwritten: old_lls is a
                    # VIEW into current_lls, so accepted rows lose their old
                    # values at the update below. Cold pair only (i2 == 0):
                    # column 0 of the slice is the cold cell whose diff is
                    # credited to log_like[0].
                    if _audit and i2 == 0:
                        _sel_h = _to_numpy(sel)
                        if _sel_h.any():
                            _bh = _to_numpy(band_inds_now[:, 0])
                            _w0 = _to_numpy(walker_inds_now[:, i2])
                            _w1 = _to_numpy(walker_inds_now[:, i1])
                            _oldh = _to_numpy(old_lls)
                            _newh = _to_numpy(new_lls)
                            for _r in np.where(_sel_h)[0]:
                                logger.info(
                                    "[TEMPER_AUDIT] accepted cold swap band=%d: "
                                    "(T0,w%d) ll %.3f -> %.3f (credit %+.3f) | "
                                    "(T1,w%d) ll %.3f -> %.3f",
                                    int(_bh[_r]), int(_w0[_r]),
                                    float(_oldh[_r, 0]), float(_newh[_r, 0]),
                                    float(_newh[_r, 0] - _oldh[_r, 0]),
                                    int(_w1[_r]),
                                    float(_oldh[_r, 1]), float(_newh[_r, 1]),
                                )

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

            with _tspan(getattr(self, "_prop_timer", None), "temper_open_close"):
                self.add_cold_chain_sources_to_residual(
                    model,
                    band_sorter,
                    extra_bool=(band_sorter.band_inds % units == bool_remainder),
                )
            if _audit:
                # Per-unit reconcile: with this parity class closed back
                # into the parent residual, the TRUE cold-walker ll delta
                # across the unit must equal the credited delta
                # (ll_change_log_temp[0] growth). A nonzero MISMATCH row
                # localizes the drift to this unit's bands and walker.
                _true_now = _to_numpy(model.analysis_container_arr.likelihood())
                _cred_now = _to_numpy(ll_change_log_temp[0].sum(axis=-1))
                _dtrue = _true_now - _audit_true_prev
                _dcred = _cred_now - _audit_cred_prev
                logger.info(
                    "[TEMPER_AUDIT] unit %d (bands %% %d == %d): true cold "
                    "delta %s | credited %s | MISMATCH %s",
                    tmp, units, bool_remainder,
                    np.array2string(_dtrue, precision=3),
                    np.array2string(_dcred, precision=3),
                    np.array2string(_dtrue - _dcred, precision=3),
                )
                _audit_true_prev = _true_now
                _audit_cred_prev = _cred_now

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

        Thin wrapper over :meth:`_window_residual_lls` on the BAND grid;
        also stores ``self._band_dof`` (per-band real dof, used to scale the
        legacy nsigma convergence tolerance).
        """
        lls, dof = self._window_residual_lls(acs, self.band_edges)
        self._band_dof = dof
        return lls

    def _window_residual_lls(self, acs, edges):
        """Cold-walker residual ll ``-1/2 <r|r>`` per frequency window.

        Returns ``(lls, dof)`` with ``lls`` a host
        ``(nwalkers, len(edges) - 1)`` array and ``dof`` the real-dof count
        per window. The parent ACA holds one AC per COLD-chain walker, so
        this is exactly the per-window null the leaf-cap convergence test
        needs. Shard-aware: each per-GPU (or per-CPU-split) slab is reduced
        on its owning device via a per-bin ll followed by a cumulative-sum
        window reduction (no per-window kernel loop).

        THE WINDOW GRID IS A PARAMETER (2026-08-15). It is called on the
        band grid for the legacy per-band diagnostics AND on the cap-cell
        grid when the cap cells are wide enough to resolve
        (:meth:`_cap_cells_resolvable`). NOTE the domain's frequency
        resolution is the hard floor: in WDM the smallest resolvable window
        is one ``layer_df`` wide, so sub-layer cap cells come back with
        EMPTY (``k1 == k0``) windows and zero dof -- which is precisely what
        :meth:`_cap_cells_resolvable` detects and what pushes the cap gate
        onto the source-attributed statistic instead.
        """
        xp = self.xp
        bs = self._basis_settings
        be = _to_numpy(edges)
        num_bands = len(be) - 1
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
        window_dof = (k1 - k0) * dof_per_bin

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
        return out, window_dof

    # ------------------------------------------------------------------
    # Leaf-cap CELL grid helpers (user design 2026-08-15)
    # ------------------------------------------------------------------
    # Every helper here short-circuits at ``cap_divisor == 1`` so the whole
    # cap machinery collapses back onto the band grid bit-identically.

    def _cap_cell_index(self, band_inds, freqs_hz):
        """Cap-cell index of sources at ``freqs_hz`` inside ``band_inds``.

        Containment (cell ``c`` belongs to band ``c // K``) makes this a
        pure per-source arithmetic lookup -- no searchsorted over a second
        edge array, and correct under BOTH band-edge modes.
        """
        if self.cap_divisor == 1 or freqs_hz is None:
            return band_inds
        xp = get_array_module(band_inds)
        sub = xp.floor(
            (freqs_hz - self._cap_band_lo[band_inds])
            / self._cap_band_step[band_inds]
        )
        sub = xp.clip(sub, 0, self.cap_divisor - 1).astype(band_inds.dtype)
        return band_inds * self.cap_divisor + sub

    def _sorter_cap_cells(self, band_sorter):
        """Per-source cap-cell index for every row of ``band_sorter``.

        ALIVE rows are what this is meaningful for: a dead row's stored
        ``freqs`` are stale, and the cell an RJ birth lands in is set by the
        DRAWN frequency (see :meth:`_run_rj_step`'s prior gate), not by this.
        Dead rows are handled through band SATURATION instead
        (:meth:`_band_saturated_flat`).
        """
        if self.cap_divisor == 1:
            return band_sorter.band_inds
        return self._cap_cell_index(band_sorter.band_inds, band_sorter.freqs)

    def _cap_flat_index(self, temp_inds, walker_inds, cap_inds):
        """Flat ``(temp, walker, cap cell)`` index (the occupancy bincount key)."""
        xp = get_array_module(cap_inds)
        return (
            (temp_inds.astype(xp.int64) * self.nwalkers + walker_inds)
            * self.num_cap_cells
            + cap_inds
        )

    def _cap_cell_counts(self, band_sorter, cap_inds=None):
        """``(ntemps*nwalkers*num_cap_cells,)`` alive-source occupancy census.

        TODO(USER, 2026-08-15, HIGH-PRIORITY CHECK): IN-MODEL DRIFT ACROSS A
        CAP-CELL BOUNDARY WITHIN A BAND IS NOT POLICED. Occupancy is
        censused here at unit open / per pick round from each source's
        CURRENT f0, but an in-model repeat block can move a source's f0
        across a cell boundary mid-unit without any cap re-check -- two
        sources can end up sharing a cell that the cap machinery believes
        holds one, and the drift is only re-censused at the NEXT
        propose. At K=8 on the uniform grid a cell is ~135 FD bins vs
        ~+-16 bins of typical source support, so this should be rare --
        but under the get_n free-frequency bands (NO LONGER one layer
        per band; widths follow 2*get_N and cells shrink accordingly)
        the margin narrows. CHECK: log a per-propose count of sources
        whose cell changed between unit open and unit close, and decide
        whether the in-model accept path needs a cell-cap veto (or
        whether post-hoc re-census suffices).
        """
        xp = get_array_module(band_sorter.band_inds)
        if cap_inds is None:
            cap_inds = self._sorter_cap_cells(band_sorter)
        flat = self._cap_flat_index(
            band_sorter.temp_inds, band_sorter.walker_inds, cap_inds
        )
        alive_cells = flat[band_sorter.inds]
        nbins = self.ntemps * self.nwalkers * self.num_cap_cells
        # cupy.bincount computes max(x) first and raises on a zero-size
        # array (the zero-leaf search start hits this on GPU).
        if alive_cells.shape[0] == 0:
            return flat, xp.zeros(nbins, dtype=xp.int64)
        return flat, xp.bincount(alive_cells, minlength=nbins)

    def _band_saturated_flat(self, counts, cap):
        """``(ntemps*nwalkers*num_bands,)`` bool: EVERY cap cell of the band full.

        The at-cap test for a DEAD row. A dead slot is tied to a band, not
        to a cap cell -- its birth draw covers the whole band -- so a birth
        is impossible only when every one of the band's cells is at
        capacity. At ``cap_divisor == 1`` this is exactly the old per-band
        ``counts >= cap``.
        """
        if self.cap_divisor == 1:
            return counts >= cap
        k = self.cap_divisor
        nb = self.num_bands
        return (
            counts.reshape(-1, nb, k) >= cap.reshape(1, nb, k)
        ).all(axis=2).reshape(-1)

    def _band_flat_index(self, temp_inds, walker_inds, band_inds):
        """Flat ``(temp, walker, band)`` index (matches ``_band_saturated_flat``)."""
        xp = get_array_module(band_inds)
        return (
            (temp_inds.astype(xp.int64) * self.nwalkers + walker_inds)
            * self.num_bands
            + band_inds
        )

    def _cap_at_cap_mask(self, band_sorter, counts, cap, flat, cap_inds):
        """Per-row at-cap mask, alive rows by CELL and dead rows by BAND.

        - alive row: is MY cap cell at capacity (drives the in-model pool
          gate -- an at-cap cell never freezes sources into the pool);
        - dead row: is EVERY cap cell of my band at capacity (drives the
          birth pick skip / staged reserve -- a birth anywhere in the band
          is impossible only then).

        At ``cap_divisor == 1`` both branches are the same expression the
        pre-2026-08-15 code used, so the mask is bit-identical.
        """
        own = counts[flat] >= cap[cap_inds]
        if self.cap_divisor == 1:
            return own
        sat = self._band_saturated_flat(counts, cap)
        band_flat = self._band_flat_index(
            band_sorter.temp_inds, band_sorter.walker_inds,
            band_sorter.band_inds,
        )
        xp = get_array_module(own)
        return xp.where(band_sorter.inds, own, sat[band_flat])

    def _cap_cells_of_band(self, band_index: int):
        """``(lo, hi)`` cap-cell index range owned by sub-band ``band_index``."""
        k = self.cap_divisor
        return band_index * k, (band_index + 1) * k

    def _mirror_band_leaf_cap(self, bi) -> None:
        """Refresh the legacy per-band ``band_leaf_cap`` from the cell caps.

        The band arrays stay written so the monitor / diag scripts keep
        working unchanged. The band value is the MAX over the band's cells
        -- the answer to "how many leaves may a single spot in this band
        hold", which is what the cap plot and the band-shutoff rule both
        ask. (SUM would report the band-total allowance, which is not what
        any existing consumer means by ``band_leaf_cap``.)
        """
        if self.cap_divisor == 1:
            return
        cell_cap = bi.get("cap_cell_leaf_cap")
        if cell_cap is None:
            return
        bi["band_leaf_cap"][:] = np.asarray(cell_cap).reshape(
            self.num_bands, self.cap_divisor
        ).max(axis=1)

    def _cap_state_arrays(self, bi):
        """``(cap, iters, best)`` for whichever grid drives the caps.

        At ``cap_divisor == 1`` these ARE the band arrays -- no cap-cell
        arrays are allocated at all, so a store written before the cap grid
        existed resumes untouched and the whole gate is bit-identical.
        """
        if self.cap_divisor == 1:
            return (
                bi["band_leaf_cap"], bi["band_cap_iters"], bi["band_best_ll"],
            )
        ensure_cap_cell_fields(bi, self.num_cap_cells)
        return (
            bi["cap_cell_leaf_cap"], bi["cap_cell_iters"],
            bi["cap_cell_best_ll"],
        )

    def _cap_cells_resolvable(self, acs) -> bool:
        """Can the RESIDUAL resolve one cap cell from the next?

        The cap-gate statistic is the residual ll per cap cell when the
        domain can actually separate the cells, and a source-attributed
        statistic when it cannot. The floor is the domain's frequency
        resolution: FD bins are ``1/Tobs`` wide (a band/8 cell at the
        production 3-month config is ~135 bins, comfortably resolved),
        but WDM's is one ``layer_df``, and the production band grid is
        ONE LAYER PER BAND -- so every K > 1 cap cell there is SUB-LAYER
        and its residual window comes back empty. This detects that from
        the window bin ranges themselves rather than assuming a domain.

        ``GB_CAP_LL_SOURCE=residual|source`` forces either branch.
        """
        forced = os.environ.get("GB_CAP_LL_SOURCE", "auto").lower()
        if forced == "residual":
            return True
        if forced == "source":
            return False
        cached = getattr(self, "_cap_resolvable_cache", None)
        if cached is not None:
            return cached
        _, dof = self._window_residual_lls(acs, self.cap_edges)
        ok = bool(np.all(np.asarray(dof) > 0))
        self._cap_resolvable_cache = ok
        logger.info(
            "%s: cap-cell ll source = %s (divisor %d; %d/%d cap cells have a "
            "non-empty residual window).",
            self.name, "residual" if ok else "source-attributed",
            self.cap_divisor, int(np.sum(np.asarray(dof) > 0)),
            int(np.size(dof)),
        )
        return ok

    def _cap_cell_lls(self, model, new_state, band_lls):
        """``(lls, dof)`` cold-walker cap-cell statistic for the cap gate.

        Two sources, picked by :meth:`_cap_cells_resolvable`:

        **residual** -- ``-1/2<r|r>`` over the cell's own frequency window,
        the EXACT quantity the per-band gate uses, just on a finer grid. No
        new kernel: the per-band reduction was already a per-bin ll plus a
        cumulative-sum window difference, so the window grid is a parameter
        (:meth:`_window_residual_lls`).

        **source-attributed** -- the sum over the cell's ALIVE COLD sources
        of their own likelihood contribution ``d_h - h_h/2``, which the GB
        kernels already produce per source (``self._sorter_dh`` / ``_hh``,
        mirrored into the sub-state's ``d_h`` / ``h_h``). Used when the cap
        cells are narrower than the domain can resolve (WDM sub-layer
        cells), where NO residual-based per-cell absolute ll exists at all.

        WHY THE ATTRIBUTION IS VALID (the project's verified orthogonality
        ruling, same bilinearity argument as the ``[GB_ORTHO_LL]`` monitor):
        two sources separated by ``|df| * Tobs >> 1`` have ``<h_i|h_j> ~ 0``,
        so likelihood contributions add. Cap cells are tens of microHz =
        hundreds of FD bins apart, so a cell's realized ll change is the sum
        of its own sources' contributions to within cross-terms that are
        ~0. Sources INSIDE one cell are exactly the ones the cap is
        throttling and their (possibly non-orthogonal) joint contribution is
        attributed to that one cell -- which is the intended behaviour, not
        an approximation.

        LIMITATION (documented, not silent): the source statistic sees only
        sources; a cell whose improvement comes from a tempering swap
        importing a better hot-rung configuration registers that change only
        through the resulting cold-chain sources, which is the same thing
        the band gate would see one iteration later.
        """
        if self._cap_cells_resolvable(model.analysis_container_arr):
            lls, dof = self._window_residual_lls(
                model.analysis_container_arr, self.cap_edges
            )
            self._cap_ll_check(lls, band_lls)
            return lls, dof
        return self._cap_cell_source_lls(new_state), np.zeros(
            self.num_cap_cells
        )

    def _cap_cell_source_lls(self, new_state):
        """``(nwalkers, num_cap_cells)`` sum of ``d_h - h_h/2`` per cap cell."""
        sub = new_state.sub_states[self.branch_name]
        branch = self._work_branch(new_state)
        # cold row of the module ladder = the joint solution
        coords = _to_numpy(branch.coords[0])           # (nw, nleaves, ndim)
        inds = _to_numpy(branch.inds[0]).astype(bool)  # (nw, nleaves)
        d_h = _to_numpy(getattr(sub, "d_h", None))
        h_h = _to_numpy(getattr(sub, "h_h", None))
        out = np.zeros((coords.shape[0], self.num_cap_cells))
        if d_h is None or h_h is None:
            return out
        contrib = np.nan_to_num(d_h - 0.5 * h_h, nan=0.0,
                                posinf=0.0, neginf=0.0)
        f0_hz = coords[..., 1] / 1e3
        be = _to_numpy(self.band_edges)
        band = np.clip(np.searchsorted(be, f0_hz, side="right") - 1,
                       0, self.num_bands - 1)
        step = (be[1:] - be[:-1]) / self.cap_divisor
        sub_i = np.clip(
            np.floor((f0_hz - be[:-1][band]) / step[band]).astype(int),
            0, self.cap_divisor - 1,
        )
        cell = band * self.cap_divisor + sub_i
        for w in range(coords.shape[0]):
            m = inds[w]
            if not m.any():
                continue
            np.add.at(out[w], cell[w][m], contrib[w][m])
        return out

    def _cold_occupancy(self, band_counts, new_state):
        """Cold-chain per-unit occupancy for the ``require_occupancy`` test."""
        if self.cap_divisor == 1:
            return _to_numpy(band_counts[0])  # (nwalkers, num_bands)
        branch = self._work_branch(new_state)
        coords = _to_numpy(branch.coords[0])
        inds = _to_numpy(branch.inds[0]).astype(bool)
        be = _to_numpy(self.band_edges)
        f0_hz = coords[..., 1] / 1e3
        band = np.clip(np.searchsorted(be, f0_hz, side="right") - 1,
                       0, self.num_bands - 1)
        step = (be[1:] - be[:-1]) / self.cap_divisor
        sub_i = np.clip(
            np.floor((f0_hz - be[:-1][band]) / step[band]).astype(int),
            0, self.cap_divisor - 1,
        )
        cell = band * self.cap_divisor + sub_i
        out = np.zeros((coords.shape[0], self.num_cap_cells), dtype=int)
        for w in range(coords.shape[0]):
            m = inds[w]
            if m.any():
                np.add.at(out[w], cell[w][m], 1)
        return out

    def _cap_ll_check(self, cell_lls, band_lls) -> None:
        """[GB_CAP_LL_CHECK] does the cap grid partition the band ll? (default OFF)

        The user's "check the LL diff in those new sub-band widths" made
        concrete and free of new kernels: sum each band's cap-cell residual
        lls and compare against the band's own residual ll. A faithful
        refinement telescopes exactly (the per-bin cumulative sum is the
        same, only the difference points move); a discrepancy means the cap
        windows do not tile their band -- e.g. the WDM boundary layer that
        adjacent windows SHARE by construction, which this reports rather
        than hides. ``GB_CAP_LL_CHECK_TOL`` sets the warning threshold.
        """
        if os.environ.get("GB_CAP_LL_CHECK", "0") != "1":
            return
        summed = cell_lls.reshape(
            cell_lls.shape[0], self.num_bands, self.cap_divisor
        ).sum(axis=2)
        diff = np.abs(summed - band_lls)
        tol = float(os.environ.get("GB_CAP_LL_CHECK_TOL", "1e-6"))
        scale = np.maximum(np.abs(band_lls), 1.0)
        rel = diff / scale
        k = int(np.argmax(rel))
        w, b = np.unravel_index(k, rel.shape)
        logger.info(
            "[GB_CAP_LL_CHECK %s] sum over %d cap cells vs band ll: max abs "
            "diff %.3e (rel %.3e) at walker %d band %d (cells %d-%d); mean "
            "abs %.3e.",
            self.name, self.cap_divisor, float(diff.max()), float(rel[w, b]),
            int(w), int(b), *self._cap_cells_of_band(int(b)),
            float(diff.mean()),
        )
        if float(rel[w, b]) > tol:
            logger.warning(
                "[GB_CAP_LL_CHECK %s] the cap-cell windows do not tile band "
                "%d: sum of cells %.6e vs band %.6e (rel %.3e > tol %.3e). "
                "The per-cell cap gate is reading a different likelihood "
                "than the per-band one.",
                self.name, int(b), float(summed[w, b]), float(band_lls[w, b]),
                float(rel[w, b]), tol,
            )

    def _update_band_leaf_caps(self, model, new_state, band_counts) -> None:
        """Advance the progressive leaf caps (once per iteration).

        Runs at the very end of ``propose`` (after the final
        ``check_ll_inject`` rebuild, so the parent residual reflects the
        accepted state). Bands increment independently; nothing waits on
        other bands. On increment the iteration counter and running best
        reset, so the next level must re-converge on its own evidence.
        Every cold walker's per-band ll is stored in
        ``band_info['band_cold_ll']`` each step, whichever gate runs, so
        any criterion can be replayed on the trace post hoc.

        DEFAULT gate (``leaf_cap_ll_improve``, on since 2026-08-12): a
        band's cap increments once its cold-chain MAX ll has failed to
        improve on the stored best by ``leaf_cap_ndim / 2`` (D/2 = 4.0
        for GBs -- the logL a genuinely new D-parameter source has to
        buy) for ``leaf_cap_min_iters`` CONSECUTIVE iterations; any
        qualifying improvement zeroes the counter.

        Legacy gate (``leaf_cap_ll_improve=False``): the cap increments
        when ALL of:

        1. ``band_cap_iters[b] >= leaf_cap_min_iters`` at the current cap;
        2. every cold walker's band residual ll sits within
           ``leaf_cap_ll_nsigma * sqrt(N_b / 2)`` of the running best
           (``N_b`` = real dof in the band -- for a whitened residual
           ``-1/2<r|r>`` fluctuates with sigma ~ sqrt(N_b/2), so this is the
           "converged up to statistical change" test);
        3. (optional, ``leaf_cap_require_occupancy``) at least one cold
           walker actually holds ``cap[b]`` leaves in the band -- an
           exhausted band with free headroom keeps its cap.

        With ``leaf_cap_iter_only`` the gate is ONLY test 1 (a fixed
        annealing schedule): the lnL-plateau and occupancy tests are
        skipped; the ``cap < nleaves_max`` guard and the log-line format
        are unchanged.
        """
        bi = new_state.sub_states[self.branch_name].band_info
        cap, iters, best = self._cap_state_arrays(bi)
        is_cells = self.cap_divisor > 1

        # The per-band residual lls are computed and stored EVERY step
        # regardless of which grid drives the caps: they are the monitor's
        # series and the auditable trace, and the legacy nsigma gate's
        # tolerance is scaled by ``self._band_dof`` which this sets.
        band_lls = self._band_residual_lls(model.analysis_container_arr)
        if ("band_cold_ll" in bi
                and bi["band_cold_ll"].shape == band_lls.shape):
            bi["band_cold_ll"][:] = band_lls

        if is_cells:
            lls, dof = self._cap_cell_lls(model, new_state, band_lls)
            if ("cap_cell_cold_ll" in bi
                    and bi["cap_cell_cold_ll"].shape == lls.shape):
                bi["cap_cell_cold_ll"][:] = lls
        else:
            lls, dof = band_lls, self._band_dof
        cur_max = lls.max(axis=0)

        if self.leaf_cap_ll_improve:
            # Coarse likelihood-based gate: a band's cap holds while the
            # cold chain keeps finding a max that beats the stored best by
            # at least D/2 -- the log-likelihood a genuinely new source of
            # D parameters has to buy to be worth admitting. Once no such
            # improvement appears for ``leaf_cap_min_iters`` consecutive
            # iterations, the band has stopped paying for its current
            # allowance and the cap increments.
            #
            # Deliberately MAX-only: it asks "is the best walker still
            # finding better fits", not "have all walkers converged
            # together" (the older nsigma test below).
            #
            # TODO(leaf-cap-min-ll): consider the MIN over cold walkers too.
            # Max-only cannot tell a band where every walker is climbing
            # from one where a single walker carries the band while the
            # rest are stuck -- the second case is a mixing problem the cap
            # will happily paper over by incrementing on schedule.
            # GHOST-INCREMENT GUARD (user ruling 2026-08-16): the patience
            # counter only runs for a cell whose max ll has improved AT
            # LEAST ONCE, mirroring ``changed_once`` in the PSD max-logL
            # search (``psdmove.py::run_move_max_likelihood``), which will
            # not count a plateau iteration until the chain has moved at
            # all.
            #
            # WHY. An EMPTY cell can never improve -- there is no source in
            # it to find a better fit for -- so under the bare counter it
            # accrued patience every iteration and promoted itself on a
            # fixed clock alongside cells doing real work. Measured on the
            # 3-month run: of ~820 empty cells only 1-16 improved per
            # iteration (0.1-2%), against 22-26% of OCCUPIED cells, and 920
            # of 1,232 cells incremented in lockstep at iteration 10. That
            # made the cap a wall-clock ratchet: protection strongest early
            # when the model is empty and needs none, weakest late when it
            # is full and confusion is worst -- backwards. Freezing the
            # never-improved cells leaves the cap where it belongs until
            # something is actually found there.
            #
            # The flag is deliberately in-memory only (a restart re-earns
            # it, pausing a cell's ramp until it shows an improvement --
            # the conservative direction), matching the band-shutoff
            # streak bookkeeping in ``_update_band_shutoff``.
            thresh = 0.5 * float(self.leaf_cap_ndim)
            improved = cur_max > (best + thresh)
            # OPT-IN (user ruling 2026-08-16): hold at GB_CAP_DIVISOR=8 with
            # the guard OFF for now. The guard and a finer cell grid are
            # COUPLED -- freezing empty cells at cap 1 while the cell is
            # still 135 FD bins wide would re-impose the 24.5% structural
            # exclusion the census measured. Default 0 reproduces today's
            # ratchet exactly; flip to 1 in the same change as K.
            if os.environ.get("GB_LEAF_CAP_REQUIRE_IMPROVEMENT", "0") != "1":
                best[:] = np.maximum(best, cur_max)
                iters[improved] = 0
                iters[~improved] += 1
                converged = iters >= self.leaf_cap_min_iters
                self._cap_ll_improved_once = None
                _skip_guard = True
            else:
                _skip_guard = False
            _seen = getattr(self, "_cap_ll_improved_once", None)
            if _seen is None or _seen.shape != improved.shape:
                _seen = np.zeros(improved.shape, dtype=bool)
            # The FIRST observation beats ``best = -inf`` trivially and is
            # not evidence of anything -- the PSD idiom guards the same way
            # with ``not np.isinf(max_logl)``. ``best`` is also reset to
            # -inf on every increment, so this re-arms per rung while the
            # flag itself stays sticky: a cell that has proven itself once
            # keeps its clock.
            if not _skip_guard:
                _seen |= improved & np.isfinite(best)
                self._cap_ll_improved_once = _seen
                best[:] = np.maximum(best, cur_max)
                iters[improved] = 0
                iters[~improved & _seen] += 1
                converged = iters >= self.leaf_cap_min_iters
        elif self.leaf_cap_iter_only:
            best[:] = np.maximum(best, cur_max)
            iters += 1
            # Iteration-only mode (see ctor): a fixed schedule -- every band
            # increments after ``leaf_cap_min_iters`` iterations at its
            # current cap, regardless of lnL plateau or occupancy.
            converged = iters >= self.leaf_cap_min_iters
        else:
            best[:] = np.maximum(best, cur_max)
            iters += 1
            tol = self.leaf_cap_ll_nsigma * np.sqrt(np.maximum(dof, 0) / 2.0)
            converged = (iters >= self.leaf_cap_min_iters) & (
                (best - lls.min(axis=0)) <= tol
            )
            if self.leaf_cap_require_occupancy:
                cold_counts = self._cold_occupancy(band_counts, new_state)
                converged &= cold_counts.max(axis=0) >= cap
        # THE PER-CELL CEILING IS THE FULL BRANCH ``nleaves_max``, NOT
        # ``nleaves_max / K`` (user intent 2026-08-15: caps SMALLER locally,
        # but the band-level TOTAL allowance never TIGHTER than today). A
        # band's total allowance therefore GROWS from ``cap`` to ``K * cap``
        # once its cells are all at the same level -- removing the
        # band-total throttle, which is the first-run watch (leaf growth /
        # memory), not a regression. Dividing the ceiling down would cap a
        # band below what it can hold today and is explicitly not what was
        # asked for.
        nleaves_max = self._work_branch(new_state).shape[2]
        converged &= cap < nleaves_max

        _unit = "cap cells" if self.cap_divisor > 1 else "bands"
        if np.any(converged):
            inc = np.where(converged)[0]
            cap[converged] += 1
            iters[converged] = 0
            best[converged] = -np.inf
            logger.info(
                f"{self.name}: leaf cap incremented for {len(inc)} {_unit} "
                f"{_compact_index_ranges(inc)} -> caps "
                f"{int(cap[inc].min())}-{int(cap[inc].max())}."
            )
        self._mirror_band_leaf_cap(bi)
        # The old tail read "at min-iters gate: {(iters < min_iters).sum()}",
        # which is TAUTOLOGICAL -- it is logged after ``iters[converged] = 0``,
        # and every non-converged cell is below the gate by definition, so it
        # printed the cell count every single iteration and carried no
        # information. Report the split that actually explains the ramp:
        # how many cells are frozen because they have never improved (the
        # ghost-increment guard) versus how many are running a live clock.
        _seen = getattr(self, "_cap_ll_improved_once", None)
        if _seen is not None and _seen.shape == iters.shape:
            _frozen = int((~_seen).sum())
            _running = int(_seen.sum())
            _tail = (f"; {_frozen} never-improved (clock frozen), "
                     f"{_running} accruing patience")
        else:
            _tail = ""
        logger.info(
            f"{self.name}: leaf caps min/max = {int(cap.min())}/{int(cap.max())}"
            f" over {len(cap)} {_unit}{_tail}."
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
        # Tempering-cadence census: every propose of this branch ticks the
        # shared counter (see _temper_cadence_fire).
        GBSpecialBase._branch_propose_counts[self.branch_name] = (
            GBSpecialBase._branch_propose_counts.get(self.branch_name, 0) + 1
        )
        # [GB_ACCEPT rj-split] per-propose class counters + the per-band
        # birth-accept tally feeding the high-f barren-band shutoff
        # (user requests 2026-08-14). Both consumed in the propose-end
        # summary block.
        self._rj_split = {} if self.is_rj_prop else None
        # [FSTAT_CTR] per-propose census of lookup-miss fallback rows (the
        # live-cap reserve rows computed per round instead of at unit open);
        # bumped in _fstat_ctr_lookup, reported in the propose-end summary.
        self._fstat_ctr_fallback_rows = 0
        # (Band-shutoff bookkeeping is occupancy-based as of 2026-08-15 —
        # measured at propose end from the cold-chain state; no per-round
        # accumulator needed.)
        # SubBandBuffer cache: one allocation per signature, units rebind in
        # place. GB_BUFFER_PERSIST=1 (default) keeps the cached buffers
        # ACROSS proposals -- construction (thousands of per-cell container
        # builds, ~16 s at full band) happens once per signature per RUN and
        # later proposals only refill/rebind. GB_BUFFER_PERSIST=0 restores
        # the July proposal-scoped contract (drop + pool sweep at every
        # proposal exit) for memory-tight multi-branch runs where the other
        # modules need the buffers' GPU memory between GB proposals.
        if (os.environ.get("GB_BUFFER_PERSIST", "1") != "1"
                or getattr(self, "_prop_buffer_cache", None) is None):
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

        # Alive-only RJ variants (replace / removal-only) are no-ops on an
        # empty model -- their alive-only BandSorter cannot be built from
        # zero sources -- so return before constructing it. The gate is
        # rj_removal_only, NOT use_prior_removal: the search BIRTH move
        # carries use_prior_removal=True (prior-judged deaths) but births
        # from dead slots, so it MUST run on the zero-leaf search start.
        # (use_prior_removal in this gate silently no-opped rj_fstat_search
        # forever on GB_MODE=search -- the 3-month run's gb_search completed
        # with 0 sources, found 2026-08-13.)
        if (
            (self.rj_replace or self.rj_removal_only)
            and not np.any(work_in.inds)
        ):
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
        self._cap_leaf_cap = None
        if self._leaf_cap_enabled and self.is_rj_prop:
            bi = state.sub_states[self.branch_name].band_info
            # The state's cap grid and the move's divisor must agree, or the
            # per-cell arrays and the per-cell indices mean different things.
            # (The resume guard in GBState catches a MISMATCHED STORE; this
            # catches a mis-wired move against a correct state.)
            _stored_cells = len(np.asarray(bi["cap_edges"])) - 1 if (
                "cap_edges" in bi) else self.num_bands
            if int(_stored_cells) != int(self.num_cap_cells):
                raise ValueError(
                    f"{self.name}: cap grid mismatch -- the move is built "
                    f"with cap_divisor={self.cap_divisor} "
                    f"({self.num_cap_cells} cap cells over "
                    f"{self.num_bands} sub-bands) but the state carries "
                    f"{_stored_cells} cap cells. Pass the SAME "
                    f"GBSettings.cap_divisor to the move builder and to "
                    f"GBState.initialize_band_information(cap_edges=...)."
                )
            ensure_leaf_cap_fields(bi, self.num_bands)
            ensure_cap_cell_fields(bi, self.num_cap_cells)
            cap_arr = self._cap_state_arrays(bi)[0]
            if np.all(cap_arr < 0):
                cap_arr[:] = int(self.leaf_cap_start)
                if self.cap_divisor > 1:
                    bi["band_leaf_cap"][:] = int(self.leaf_cap_start)
                logger.info(
                    f"{self.name}: armed leaf cap at "
                    f"{int(self.leaf_cap_start)} for {len(cap_arr)} "
                    + ("cap cells " if self.cap_divisor > 1 else "bands ")
                    + f"(divisor {self.cap_divisor} over "
                    f"{self.num_bands} sub-bands)."
                )
            self._cap_leaf_cap = cap_arr
            # ``_band_leaf_cap`` stays the ARMED flag + the band-resolution
            # mirror the shutoff rule and the monitor read; every cap
            # decision reads ``_cap_leaf_cap``.
            self._band_leaf_cap = bi["band_leaf_cap"]
            self._mirror_band_leaf_cap(bi)

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
        # rj_replace joins use_prior_removal here: both act only on ALIVE
        # leaves, so the sorter carries just the alive sources (no dead-slot
        # pre-draws; the replacement draw happens per pick in
        # _run_replace_step). The sorter's +logpdf factors then carry the
        # removed/replaced source's death-side proposal term.
        if self.use_prior_removal or self.rj_replace:  # TODO: make this stronger?
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
                opt_snr_rej_samp_limit=self.opt_snr_rej_samp_limit,
                snr_rej_detected=self.snr_rej_detected,
            )

        # Cold-chain friend table for the group-stretch half of the in-model
        # mix (rebuilt every proposal; cheap sort of the cold-chain f0s).
        self._infomat_wdm_logged = False
        # The cold-chain friend / info-matrix tables are NOT built here: they
        # are built lazily at the first in-model block of the proposal, which
        # is after that iteration's first RJ step, so they see the post-RJ
        # source population (see _ensure_proposal_tables). This flag makes the
        # per-proposal indexing happen exactly once per sorter.
        self._tables_indexed = False

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

        self._debug_sync_all_devices(model)
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

        # Safety for the temper relocation (user ruling 2026-08-14: swaps
        # run inside rj_prior_removal, not the fstat-birth move): a move
        # designated as the swap carrier whose gate can never fire must
        # SAY so -- a silently frozen temperature ladder is a correctness
        # bug, not a tuning choice.
        if (
            self.run_swaps
            and (self.temperature_control is None or self.ntemps <= 1)
            and not getattr(self, "_run_swaps_skip_warned", False)
        ):
            self._run_swaps_skip_warned = True
            logger.warning(
                "%s: run_swaps=True but tempering cannot run "
                "(temperature_control=%s, ntemps=%d) -- band-temperature "
                "swaps are NOT happening through this move.",
                self.name,
                "set" if self.temperature_control is not None else "None",
                self.ntemps,
            )
        if (
            self.temperature_control is not None
            and self.time % 1 == 0
            and self.ntemps > 1
            and (self.is_rj_prop or self.swap_on_in_model)
            and self.run_swaps
            # cadence LAST: only consumes the per-branch budget when every
            # other gate already passed (see _temper_cadence_fire).
            and self._temper_cadence_fire()
        ):
            st_temp = time.perf_counter()
            with tm.span("ll_checks"):
                ll_before1 = model.analysis_container_arr.likelihood()

            with tm.span("run_tempering"):
                ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed = self.run_tempering(
                    model, new_state, band_sorter, band_temps
                )

            new_state.log_like[0] += _to_numpy(ll_change_sum_temp[0])

            self._debug_sync_all_devices(model)
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

        # ACCEPTANCE RATES for this propose, split RJ vs in-model, cold chain
        # (temp 0) and all-temperature. Nothing logged them before, which made
        # every proposal-machinery A/B unjudgeable -- exact-vs-borrowed info
        # matrices, whether the pure in_model move still earns its place, a
        # changed jump scale. Pure logging; counters already exist.
        try:
            _pc, _ac = _to_numpy(prop_counts), _to_numpy(acc_counts)

            def _rate(row, cold_only):
                pr = _pc[row][0] if cold_only else _pc[row]
                ac = _ac[row][0] if cold_only else _ac[row]
                tot = float(pr.sum())
                return (float(ac.sum()) / tot if tot > 0 else float("nan"), tot)

            (rj_c, rj_cn), (rj_a, rj_an) = _rate(0, True), _rate(0, False)
            (im_c, im_cn), (im_a, im_an) = _rate(1, True), _rate(1, False)
            logger.info(
                "[GB_ACCEPT %s] rj cold %.4f (n=%.0f) all %.4f (n=%.0f) | "
                "in-model cold %.4f (n=%.0f) all %.4f (n=%.0f)",
                self.name, rj_c, rj_cn, rj_a, rj_an, im_c, im_cn, im_a, im_an,
            )
            # PER-WALKER cold in-model decomposition (shard-bug hunt
            # 2026-08-12): on 2 GPUs the aggregate cold rate halves vs
            # 1 GPU. (a, a, ~0, ~0) convicts the WALKER-sharded parent
            # side (shard-1 residual rows); a uniform a/2 convicts the
            # BAND-sharded cell scoring side (device-1 band cells).
            with np.errstate(invalid="ignore", divide="ignore"):
                _im_w = _ac[1][0].sum(axis=-1) / _pc[1][0].sum(axis=-1)
            logger.info(
                "[GB_ACCEPT %s] in-model cold PER WALKER: %s",
                self.name, np.array2string(_im_w, precision=3),
            )
        except Exception as exc:  # never break a propose for a log line
            logger.debug("[GB_ACCEPT %s] skipped: %r", self.name, exc)

        # [FSTAT_CTR] propose-end total of lookup-miss fallback rows (the
        # per-unit precompute lines carry the running count; this is the
        # per-propose sum for the job-195 rj_fstat_centers growth hunt).
        if (self.is_rj_prop and getattr(self, "rj_fstat_dist_birth", False)
                and not self.rj_replace):
            _tbl = self._fstat_ctr_table_active()
            logger.info(
                "[FSTAT_CTR %s] propose total: mode=%s (%s), "
                "fallback-computed rows %d", self.name,
                self._fstat_ctr_mode(),
                f"{int(_tbl['f0_mHz'].shape[0])}-node table"
                if _tbl is not None else "per-unit hoist",
                int(getattr(self, "_fstat_ctr_fallback_rows", 0)),
            )

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
        (VGB is pure stretch: fixed-dimensional, no info-matrix branch.)

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
        self._last_im_kind = "stretch"
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


#: Process-local cache of fitted F-stat birth grids, keyed by epoch cache
#: directory -> ``(container, epoch, n_peaks)``. Lets every move sharing a
#: fit dir reuse the FIRST one's result with no refit and no npz reload.
#: Cleared only by process exit; the on-disk epoch caches are the
#: cross-process equivalent.
_FSTAT_GRID_REGISTRY: dict = {}

#: Process-local cache of epoch F-stat CENTER tables, keyed by epoch cache
#: directory -> the device-resident table dict (or ``None`` when that epoch
#: has no drawable f0 support). Same sharing contract as
#: :data:`_FSTAT_GRID_REGISTRY`: whichever move installs the epoch first pays
#: the sweep, the rest take the identical table.
_FSTAT_CTR_TABLE_REGISTRY: dict = {}


class GBSpecialRJFStatGridMove(GBSpecialRJPriorMove):
    """RJ birth move that FITS its own F-stat grid inside ``setup()``.

    The offline prep (``scripts/fstat_proposal/plot_fstat_proposal_mojito.py``
    -> ``*_comb.npz`` + ``*_peaks_stacked.npz``) becomes unnecessary: on the
    first hit this move runs the full comb scan -> peak selection -> stage-B
    grid build against the **current residual**, installs the result as its
    RJ birth proposal, and then lets ``propose()`` continue normally. The
    compute cores are the library ones
    (:mod:`lisatools.sampling.fstat_gridfit`) -- nothing is reimplemented
    here; this class only decides *when* to fit and wires the result in.

    ``setup()`` runs once per ``propose()`` (``GBSpecialBase.setup`` is
    called immediately before ``rj_proposal_distribution`` is read), so the
    decision logic below is what keeps the expensive path off every hit:

    * ``fstat_refit_every <= 0`` (default): fit exactly once, ever.
    * otherwise: refit when ``num_proposals`` has advanced that many hits
      since the last fit.

    Each fit gets its own ``epoch_<k>`` cache directory, and the sweep
    checkpoints are salted with the epoch, so a mid-fit death resumes
    exactly where it stopped and a later epoch can never resume an earlier
    epoch's rows. A prebuilt offline grid drops in as ``epoch_0000/``.
    """

    def __init__(self, *args, fstat_fit_dir: str = "",
                 fstat_refit_every: int = 0,
                 fstat_fit_kwargs: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not fstat_fit_dir:
            raise ValueError(
                "GBSpecialRJFStatGridMove needs fstat_fit_dir (the root for "
                "its epoch_<k> grid caches)."
            )
        self.fstat_fit_dir = str(fstat_fit_dir)
        self.fstat_refit_every = int(fstat_refit_every)
        self.fstat_fit_kwargs = dict(fstat_fit_kwargs or {})
        self._fstat_epoch = None
        self._fstat_last_fit_hit = -1
        # Epoch F-stat center table (GB_FSTAT_CTR_MODE=epoch); installed
        # alongside the birth grids, see _install_ctr_table.
        self._fstat_ctr_table = None

    # ---- epoch bookkeeping -------------------------------------------------

    @property
    def _fstat_root(self) -> str:
        """Where this move's epochs live.

        SHARED by default (``<fit_dir>/shared``): several moves in one run
        want the SAME birth grid -- the search cycle's ``rj_prior_search``
        and the PE ``rj_prior`` both score births against the same residual,
        so making each fit its own copy would pay the (potentially
        hours-scale) sweep once per move for an identical answer. Whichever
        move's ``setup()`` fires first does the fit; the rest pick it up
        from :data:`_FSTAT_GRID_REGISTRY` (same process, no disk round trip)
        or from the epoch's npz caches (fresh process / restart).

        ``GB_FSTAT_FIT_PER_MOVE=1`` restores per-move grids for the case
        where two moves genuinely need different ones.
        """
        if os.environ.get("GB_FSTAT_FIT_PER_MOVE", "0") == "1":
            return os.path.join(self.fstat_fit_dir, self.name)
        return os.path.join(self.fstat_fit_dir, "shared")

    def _epoch_dir(self, k: int) -> str:
        return os.path.join(self._fstat_root, f"epoch_{k:04d}")

    @staticmethod
    def _epoch_complete(d: str) -> bool:
        """An epoch is done when its stage-B npz OR its manifest exists.

        The manifest covers the legitimate zero-peak case, where no
        ``*_peaks_stacked.npz`` is ever written but the fit did run and must
        not be repeated forever.
        """
        from lisatools.sampling.fstat_gridfit import GRID_BASENAME

        return (os.path.exists(os.path.join(
                    d, GRID_BASENAME.replace(".npz", "_peaks_stacked.npz")))
                or os.path.exists(os.path.join(d, "DONE.json")))

    def _latest_epoch(self):
        root = self._fstat_root
        if not os.path.isdir(root):
            return None
        ks = sorted(int(n.split("_")[1]) for n in os.listdir(root)
                    if n.startswith("epoch_") and n[6:].isdigit())
        return ks[-1] if ks else None

    def _epoch_band_grid_stale(self, d: str) -> bool:
        """True when epoch dir ``d`` holds grids fitted on DIFFERENT band edges.

        The stage-B / comb npz caches store the ``band_edges`` the fit ran
        against; per-peak ``band_idx`` labels are indices into that grid, so
        any band-edge change (GB_BAND_EDGES_MODE / GB_BAND_TARGET_COUNT /
        GB_SUBBAND_DIVISOR) invalidates the whole epoch. A cache with no
        band metadata at all also counts as stale (unverifiable). Missing
        caches (e.g. a DONE.json-only zero-peak epoch) are NOT stale -- they
        carry no band-indexed state.
        """
        from lisatools.sampling.fstat_gridfit import check_cached_band_grid

        try:
            check_cached_band_grid(d, _to_numpy(self.band_edges))
        except ValueError as exc:
            logger.warning(
                "%s: F-stat epoch cache %s is STALE under the current band "
                "grid (%s); forcing a fresh-epoch refit.", self.name, d, exc)
            return True
        return False

    def _fstat_fit_decision(self):
        """Pure decision helper -> ``(action, epoch)``.

        ``action`` is one of ``"skip"`` (keep the installed proposal),
        ``"load"`` (a complete epoch exists on disk) or ``"fit"``.
        Factored out so the state machine is table-testable without a
        sampler.
        """
        if self.rj_proposal_distribution is not None:
            if self.fstat_refit_every <= 0:
                return "skip", self._fstat_epoch
            if (self.num_proposals - self._fstat_last_fit_hit
                    < self.fstat_refit_every):
                return "skip", self._fstat_epoch
            return "fit", (self._fstat_epoch or 0) + 1
        k_latest = self._latest_epoch()
        if k_latest is None:
            return "fit", 0
        if self._epoch_band_grid_stale(self._epoch_dir(k_latest)):
            # Fitted against a different band grid (band-edge knobs changed
            # across a resume): NEVER load it, and do NOT resume its sweep
            # checkpoints either (their fingerprint does not cover the band
            # grid) -- start a fresh epoch directory.
            return "fit", k_latest + 1
        if self._epoch_complete(self._epoch_dir(k_latest)):
            return "load", k_latest
        # Mid-fit death: resume the SAME epoch (its checkpoints are live).
        return "fit", k_latest

    # ---- the fit -----------------------------------------------------------

    def _fstat_call(self, model, walker_ref):
        """The injectable kernel entry the library sweeps drive.

        Default: same routing as :meth:`_fstat_NM` -- the sig-het wrapper
        unwrap and the multi-shard route are both load-bearing, so this
        reuses that method rather than re-deriving the comp.

        ``FSTAT_USE_SIGHET=1`` (opt-in; default stays the chunked path)
        swaps in the signal-het shared-reference F-stat: the comb/stage-B
        sweeps score through
        :func:`lisatools.sampling.fstat_gridfit.build_sighet_call_fstat`,
        which builds bucketed heterodyne references against the reference
        walker's residual once and evaluates every candidate through the
        ``gb_signal_het_fstat_get_ll`` kernel. Requires the GB comp to be a
        ``GBSignalHetComputations`` (GB_SIGHET_INMODEL=1 wiring) built with
        ``v4_knots > 0`` and a WDM basis; anything else falls back to the
        chunked path with a warning. Multi-shard holders go through
        :meth:`_RoutedBandEngine.route_sighet_fstat`, which pins the whole
        scorer to the reference walker's shard on a device-local comp
        replica; ``FSTAT_SIGHET_MULTIDEV=1`` (OPT-IN, default off pending
        the on-GPU parity gate; ``=check`` adds a pinned shadow compare)
        fans candidate batches out over ALL run devices instead.
        Single-shard holders pass through unchanged.
        """
        # Default ON (2026-08-12 user ruling): the sig-het shared-reference
        # F-stat is the production scorer; FSTAT_USE_SIGHET=0 restores the
        # chunked path (and non-WDM / non-sig-het comps fall back with a
        # warning below either way).
        if os.environ.get("FSTAT_USE_SIGHET", "1") == "1":
            sig_comp = self.gb_wdm_comp
            holder = model.analysis_container_arr
            if not hasattr(sig_comp, "setup_fstat_references"):
                logger.warning(
                    "%s: FSTAT_USE_SIGHET=1 but the GB comp has no sig-het "
                    "F-stat surface (need GB_SIGHET_INMODEL=1 with "
                    "v4_knots set); falling back to the chunked F-stat.",
                    self.name)
            elif isinstance(self._basis_settings, FDSettings):
                logger.warning(
                    "%s: FSTAT_USE_SIGHET=1 is WDM-only; falling back to "
                    "the chunked F-stat.", self.name)
            else:
                band_edges = _to_numpy(self.band_edges)
                return _RoutedBandEngine.route_sighet_fstat(
                    sig_comp, holder, xp=self.xp,
                    Tobs=float(self._basis_settings.Tobs),
                    f0_lims_hz=(float(band_edges[0]),
                                float(band_edges[-1])),
                    data_index=int(walker_ref),
                    noise_index=int(walker_ref))
        return lambda params: self._fstat_NM(model, params, walker_ref)

    def _run_fstat_fit(self, model, k: int):
        from lisatools.sampling.fstat_gridfit import run_fstat_grid_fit

        cache_dir = self._epoch_dir(k)
        os.makedirs(cache_dir, exist_ok=True)
        walker_ref = self._fstat_reference_walker(model)
        # Auditability: the epoch line carries the reference walker's total
        # lnL (residual+PSD combination) alongside its index, so a run log
        # shows WHICH state each epoch's grid was fitted against -- "same
        # peaks after a refit" is only diagnosable with this visible.
        try:
            _lls = _to_numpy(model.analysis_container_arr.likelihood())
            _ll_ref, _ll_spread = (float(_lls[walker_ref]),
                                   float(_lls.max() - _lls.min()))
        except Exception:
            _ll_ref, _ll_spread = float("nan"), float("nan")
        band_edges = _to_numpy(self.band_edges)
        # f0_lims convention (gb.py): the interior span, band_edges[1:-1].
        f0_lims = (float(band_edges[1]), float(band_edges[-2]))
        mc_lims = self.fstat_fit_kwargs.get("mc_lims") or [0.001, 1.0]
        t0 = time.perf_counter()
        logger.info("%s: F-stat grid fit epoch %d starting (walker_ref=%d, "
                    "lnL=%.3f, cold-walker lnL spread=%.3f, cache %s)",
                    self.name, k, walker_ref, _ll_ref, _ll_spread, cache_dir)
        stacked, n_peaks = run_fstat_grid_fit(
            self._fstat_call(model, walker_ref),
            xp=self.xp,
            Tobs=float(self._basis_settings.Tobs),
            band_edges_hz=band_edges,
            f0_lims_hz=f0_lims,
            mc_lims=mc_lims,
            cache_dir=cache_dir,
            fingerprint_extra=f"|epoch={k}",
        )
        wall = time.perf_counter() - t0
        logger.info("%s: F-stat grid fit epoch %d done in %.1fs (%d peaks)",
                    self.name, k, wall, n_peaks)
        try:
            with open(os.path.join(cache_dir, "DONE.json"), "w") as f:
                json.dump(dict(epoch=k, walker_ref=int(walker_ref),
                               n_peaks=int(n_peaks), wall_seconds=wall,
                               num_proposals=int(self.num_proposals)), f)
        except OSError as exc:  # manifest is bookkeeping, never fatal
            logger.warning("%s: could not write DONE.json (%r)", self.name, exc)
        return stacked, n_peaks

    # Fields the propose-time lookup reads; the rest of the npz (node Mc /
    # sky provenance) stays on disk rather than eating device memory.
    _CTR_TABLE_DEVICE_FIELDS = (
        "f0_mHz", "phi0", "iota", "psi", "ln_A_max", "sigma_base", "ln_snr")

    def _install_ctr_table(self, k: int, model=None):
        """Load or build epoch ``k``'s F-stat center table and install it.

        The USER RULING's "compute them once" step: ONE batched sweep over
        the birth proposal's drawable f0 support
        (:func:`lisatools.sampling.fstat_gridfit.enumerate_center_nodes` --
        every peak-box f0 node at its own F-stat argmax in (Mc, sky), plus
        the comb nodes at their scan-best sky), scored through the SAME
        ``call_fstat`` the grids were fitted with, against the SAME reference
        walker's residual snapshot. Persisted as ``fstat_centers.npz`` in the
        epoch dir, so a restart that loads a complete epoch loads the centers
        with it (in milliseconds, and WITHOUT building an F-stat scorer); an
        epoch that predates the table (or an offline grid dropped into
        ``epoch_0000``) rebuilds it here, against the residual as it stands
        at that moment.

        No-ops under ``GB_FSTAT_CTR_MODE=unit``. Leaves ``_fstat_ctr_table``
        ``None`` when the epoch has no drawable support at all — the move
        then falls back to the per-unit hoist.
        """
        if self._fstat_ctr_mode() != "epoch":
            self._fstat_ctr_table = None
            return
        key = self._epoch_dir(k)
        if key in _FSTAT_CTR_TABLE_REGISTRY:
            self._fstat_ctr_table = _FSTAT_CTR_TABLE_REGISTRY[key]
            return

        from lisatools.sampling.fstat_gridfit import (
            CENTER_TABLE_BASENAME,
            build_fstat_center_table,
        )

        # Only build the scorer when a sweep is actually needed: under
        # FSTAT_USE_SIGHET the call is a bucketed shared-reference build, far
        # too expensive to pay on the checkpoint-load path.
        need_sweep = not os.path.exists(
            os.path.join(key, CENTER_TABLE_BASENAME))
        call = (self._fstat_call(model, self._fstat_reference_walker(model))
                if (need_sweep and model is not None) else None)
        t0 = time.perf_counter()
        host = build_fstat_center_table(
            call, cache_dir=key, xp=self.xp,
            mc_lims=self.fstat_fit_kwargs.get("mc_lims") or [0.001, 1.0],
            max_nodes=int(os.environ.get("GB_FSTAT_CTR_MAX_NODES", "1000000")),
        )
        table = None
        if host is not None:
            # Device residency: setup() runs inside the move's own propose,
            # so self.xp lands these on the run's current device (same
            # contract the buffers rely on).
            table = {name: self.xp.asarray(host[name])
                     for name in self._CTR_TABLE_DEVICE_FIELDS}
            logger.info(
                "[FSTAT_CTR %s] epoch %d center table: %d nodes, f0 "
                "%.5f-%.5f mHz, smear %.2f, ready in %.1fs", self.name, k,
                int(host["f0_mHz"].size), float(host["f0_mHz"][0]),
                float(host["f0_mHz"][-1]), self._fstat_ctr_smear(),
                time.perf_counter() - t0)
        else:
            logger.warning(
                "%s: epoch %d has no F-stat center table; the RJ distance "
                "birth falls back to the per-unit hoist.", self.name, k)
        self._fstat_ctr_table = table
        _FSTAT_CTR_TABLE_REGISTRY[key] = table

    def _install(self, k: int, stacked=None, n_peaks=None):
        from lisatools.sampling.fstat_gridfit import build_gb_birth_distribution

        # New epoch container -> re-discover the census surface.
        self._stacked_census_obj = "unset"

        kw = self.fstat_fit_kwargs
        container = build_gb_birth_distribution(
            cache_dir=self._epoch_dir(k),
            mc_lims=kw.get("mc_lims") or [0.001, 1.0],
            A_lims=kw.get("A_lims"),
            dist_lims=kw.get("dist_lims"),
            fdot_astro_ratio_max=kw.get("fdot_astro_ratio_max"),
            use_cupy=self.backend.uses_cupy,
            stacked_live=stacked,
            # loud last-line refusal of grids fitted on a different band
            # grid (band_idx labels would silently re-label otherwise)
            expected_band_edges=_to_numpy(self.band_edges),
        )
        if container is None:
            # Zero peaks (or a stage that produced nothing): fall back to the
            # prior so births still happen, and leave the epoch COMPLETE so
            # this does not turn into a refit loop. ``priors`` / ``gpu_priors``
            # are ALREADY branch-keyed dicts (recipe.py builds them as
            # ``{"gb": ...}`` and hands them to the removal move unwrapped),
            # so they are assigned straight through -- re-wrapping them would
            # nest a dict where a distribution is expected.
            logger.warning(
                "%s: F-stat fit epoch %d produced no birth distribution "
                "(%s peaks); falling back to the prior for births.",
                self.name, k, n_peaks,
            )
            self.rj_proposal_distribution = (
                self.priors if not self.backend.uses_cuda else self.gpu_priors
            )
        else:
            self.rj_proposal_distribution = {self.branch_name: container}
        self._fstat_epoch = k
        self._fstat_last_fit_hit = int(self.num_proposals)
        # Publish for the other moves sharing this fit dir (see
        # :data:`_FSTAT_GRID_REGISTRY`). Store the assembled
        # rj_proposal_distribution, so the prior-fallback case is shared too
        # and a second move cannot re-run a fit that legitimately found
        # nothing.
        _FSTAT_GRID_REGISTRY[self._epoch_dir(k)] = (
            self.rj_proposal_distribution, k, n_peaks,
        )

    def setup(self, model, branches):
        action, k = self._fstat_fit_decision()
        if action == "skip":
            return

        # Cross-move reuse: another move sharing this fit dir may already
        # have built (or loaded) this exact epoch in THIS process. Take its
        # container verbatim -- no refit, no npz reload.
        if action in ("load", "fit"):
            hit = _FSTAT_GRID_REGISTRY.get(self._epoch_dir(k))
            if hit is not None:
                container, epoch, n_peaks = hit
                logger.info(
                    "%s: reusing the F-stat birth grid already fitted this "
                    "process (epoch %d, %s peaks) -- no refit.",
                    self.name, epoch, n_peaks,
                )
                self.rj_proposal_distribution = container
                self._fstat_epoch = epoch
                self._fstat_last_fit_hit = int(self.num_proposals)
                self._install_ctr_table(epoch, model=model)
                return

        if action == "load":
            logger.info("%s: loading complete F-stat grid epoch %d from %s",
                        self.name, k, self._epoch_dir(k))
            self._install(k)
            self._install_ctr_table(k, model=model)
            return
        stacked, n_peaks = self._run_fstat_fit(model, k)
        self._install(k, stacked=stacked, n_peaks=n_peaks)
        self._install_ctr_table(k, model=model)


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

    # 2026-08-12 USER RULING: the legacy GBGPU FD computations
    # (get_fstat_ll / get_ll) must never score a non-FD residual -- the
    # default GB RJ stack in BOTH modes is the F-stat grid birth move
    # (GBSpecialRJFStatGridMove, sig-het scorer) + prior RJ, and the
    # serial-MCMC / refit moves that route through here are legacy
    # FD-only surfaces. Fail LOUDLY instead of scoring WDM coefficients
    # through an FD kernel.
    _settings = getattr(acs, "settings", None)
    if _settings is not None and not isinstance(_settings, FDSettings):
        raise ValueError(
            "para_log_like routes through the legacy GBGPU FD computations "
            f"(get_fstat_ll/get_ll), but the run basis is "
            f"{type(_settings).__name__}. These kernels are FD-only and must "
            "not appear in WDM runs (2026-08-12 ruling): use the F-stat grid "
            "birth move (rj_prior / rj_prior_search) + prior RJ instead."
        )

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