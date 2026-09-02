"""Band-level infrastructure for the GB special moves.

This module owns the sub-band machinery that the GB proposal moves in
:mod:`gbspecialstretch` drive:

* :func:`pack_special_index` / :func:`unpack_special_index` -- the single
  home of the ``(temp * nwalkers + walker) * 1e6 + band`` encoding used to
  identify a (temperature, walker, band) cell of the ensemble.
* :class:`SubBandBuffer` -- the per-cell residual / PSD / template scratch
  buffer. It **is** an :class:`~lisatools.analysiscontainer.AnalysisContainerArray`
  (one :class:`~lisatools.analysiscontainer.AnalysisContainer` per active
  cell) extended with fast GB source injection/removal through a
  :class:`~lisatools.globalfit.moves.gb_likelihood.BandLikelihoodEngine`.
  ``Buffer`` is kept as a back-compat alias.
* :class:`BandSorter` -- flat per-source view of the eryn GB branch
  (coords / inds / temp / walker / leaf / band index arrays) with subset
  and RJ pre-draw machinery.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from types import ModuleType
from typing import Optional, Tuple, Union

import numpy as np
import numpy

try:
    import cupy as cp
    import cupy

    gpu_available = True
except ModuleNotFoundError:
    import numpy as cp

    gpu_available = False

from eryn.state import Branch
from eryn.utils import TransformContainer

from ...analysiscontainer import (
    AnalysisContainer,
    AnalysisContainerArray,
    BandView,
    band_gpu_assignment,
    shard_lookup_maps,
)
from ...domains import DomainSettingsBase, FDSettings, WDMSettings
from ...sensitivity import SensitivityMatrixBase
from ...utils.device import current_device, device_context
from ...utils.parallelbase import LISAToolsParallelModule
from ...utils.utility import asnumpy, get_array_module

__all__ = [
    "pack_special_index",
    "unpack_special_index",
    "SubBandBuffer",
    "Buffer",
    "BandSorter",
    "BandScheduler",
]

logger = logging.getLogger(__name__)

_to_numpy = asnumpy

# Encoding base for the band part of a special index. Bands live in
# ``[0, _SPECIAL_INDEX_BASE)``; everything above encodes (temp, walker).
_SPECIAL_INDEX_BASE = int(1e6)

# Task-b default WDM-layer leakage half-width for auto-sized per-band slabs.
# A near-monochromatic GB's WDM energy is captured by the chunked-het kernel
# over ``m_floor +/- m_band_half_width`` (default 1 -> 3 layers), which alone
# reaches median mm5 ~1e-9; the canonical mm5 band spans 5 layers. Beyond +/-2
# layers the energy is < ~1e-7 for the recommended Tukey window (alpha
# 0.01-0.05), so 2 neighbor layers each side covers the leakage conservatively.
# ``scripts/diagnostics/check_wdm_band_slab.py`` measures this directly.
_WDM_SLAB_LEAKAGE_LAYERS = 2


def fstat_nm_lane_bounds(n, lanes, weights=None):
    """Contiguous row-lane boundaries for the multi-device F-stat scorer.

    Returns ``lanes + 1`` ascending ints; lane ``i`` scores rows
    ``[bounds[i], bounds[i+1])``. Lanes are CONTIGUOUS so the merge stays
    a pure permutation into disjoint host row ranges.

    ``weights=None`` (the default, and any unusable spec) reproduces
    ``(n * arange(lanes + 1)) // lanes`` EXACTLY -- the historical split.
    Bit-identity matters here: the boundaries decide which device scores
    which row, so an unset knob must not perturb a production run.

    ``weights`` is a list of positive ints or a comma-separated string
    (``GB_FSTAT_NM_LANE_WEIGHTS``, e.g. ``"3,1"``). Rows are apportioned
    by INTEGER cumulative arithmetic ``(n * cw[i]) // cw[-1]``, which is
    what makes the equal-weight case collapse onto the legacy formula
    instead of drifting by a row through float rounding.

    WHY (measured, 2026-08-29 3-month v7 restart): the run's two H100 NVLs
    are not interchangeable -- GPU0 sat at 39.1% mean utilisation with a
    90,698 MiB peak, GPU1 at 72.1% with 70,020 MiB -- while the split was
    exactly 50/50 and every one of ~1,409 joins per propose waited on the
    slower lane. A weight lets the operator move rows toward the idle card
    without touching the kernel or the math.

    Anything malformed (wrong length, non-integer, negative, all-zero)
    falls back to the equal split: this is operator input read on a
    production node, and a typo must not stop a run.
    """
    lanes = int(lanes)
    n = int(n)
    equal = (n * np.arange(lanes + 1)) // lanes
    if weights is None:
        return equal
    if isinstance(weights, str):
        spec = weights.strip()
        if not spec:
            return equal
        parts = [p.strip() for p in spec.split(",")]
        try:
            w = [int(p) for p in parts]
        except ValueError:
            return equal
    else:
        try:
            w = [int(x) for x in weights]
        except (TypeError, ValueError):
            return equal
        if any(float(a) != float(b) for a, b in zip(w, weights)):
            return equal
    if len(w) != lanes or any(x < 0 for x in w) or sum(w) <= 0:
        return equal
    cw = np.concatenate([[0], np.cumsum(np.asarray(w, dtype=np.int64))])
    return (n * cw) // int(cw[-1])


def _band_window_strict() -> bool:
    """``GB_BAND_WINDOW_STRICT`` (default ``"0"`` -- today's behaviour).

    OFF (default) reproduces the legacy expression to the bit. ON makes
    :func:`rj_band_window` return the EXACT sub-band, so the only N/4
    widening left anywhere is the in-model bin gate in
    ``_run_in_model_repeats`` -- which is already expressed in the MOVE's
    df. See that function's docstring for why this matters.
    """
    return os.environ.get("GB_BAND_WINDOW_STRICT", "0") == "1"


def rj_band_window(band_edges, band_N_vals, band_inds, df, is_rj):
    """``[lower, upper]`` frequency limits (Hz) of each slot's band window.

    This is the array the RJ support gate compares a birth's f0 against
    (``curr_logp[(~alive) & out_of_band] = -inf`` in
    ``_run_rj_step``), and -- divided by the MOVE's df -- the ``lo_s`` /
    ``hi_s`` the in-model bin gate widens by ``n4_s``.

    ⚠ A UNIT COLLISION LIVES IN THE LEGACY BRANCH (2026-08-29). The
    ``N/4`` widening below is computed in ``df`` = the BUFFER's df, which
    on the WDM path is ``layer_df`` (see :attr:`SubBandBuffer.df`). Every
    consumer then divides by the MOVE's df, which on WDM is ``1/Tobs``
    (``GBSpecialBase._configure_domain``). ``band_N_vals`` is a count of
    FD bins at ``1/Tobs`` (it comes from ``get_N(..., Tobs, ...)``), so
    pairing it with ``layer_df`` overstates the widening by exactly
    ``layer_df * Tobs == Nt/2``. On the v7 3-month grid that is 1080x:
    an intended 128-bin (N=512) margin becomes 138,240 bins = 1024
    sub-bands, i.e. wider than the entire 3-21 mHz analysis band, so the
    window is effectively unbounded.

    ``GB_BAND_WINDOW_STRICT=1`` removes the widening entirely rather
    than re-deriving it in move units, because that is what the user's
    contract asks for (2026-08-29): "The cap and RJ apply within the
    sub-band limits only, not N/4 outside. ... We want in-model to allow
    movement across the band edge up to N/4 outside." With the window
    strict, the in-model gate's own ``n4_s`` -- already in move-df bins
    -- supplies that N/4 and nothing double-counts it.

    Args:
        band_edges: Ascending band-edge frequencies (Hz).
        band_N_vals: Per-band FD sample counts (bins at ``1/Tobs``).
        band_inds: Band index of each slot.
        df: The BUFFER's df (FD bin width, or ``layer_df`` on WDM).
        is_rj: Truthy on RJ-provenance buffers (the sorter's
            ``rj_prop``), which are the only ones ever widened.

    Returns:
        ``[lower_f_lim, higher_f_lim]``, fresh arrays in both branches.
    """
    lower = band_edges[band_inds]
    upper = band_edges[band_inds + 1]
    if not is_rj or _band_window_strict():
        return [lower, upper]
    widen = band_N_vals[band_inds] * df / 4
    return [lower - widen, upper + widen]


def band_support_halfwidths(
    band_edges, Tobs: float, *, oversample: int = 4, amp: float = 1e-30
) -> np.ndarray:
    """Edge-source half-supports ``s(f_hi_b) = get_N(f_hi_b) / Tobs`` (Hz).

    A single GB chunked-heterodyne source generates an initial
    FREQUENCY-DOMAIN heterodyned waveform whose size is set by the
    installed :func:`gbgpu.utils.utility.get_N` (the same function the
    production FD/chunked-het paths use to size per-source windows --
    never reimplemented here). A source at frequency ``f`` can therefore
    carry FD support reaching ``get_N(f) * df`` (``df = 1/Tobs``) beyond
    ``f`` on each side; a source at a band's TOP edge is the worst case
    for that band, so per band the half-support (overhang) is evaluated
    at ``f_hi_b``. This array is the single source of truth for the
    leakage bookkeeping checks (separation guard + slab coverage).
    """
    from gbgpu.utils.utility import get_N

    edges = np.asarray(asnumpy(band_edges), dtype=float)
    df = 1.0 / float(Tobs)
    return np.asarray([
        float(get_N(amp, f, Tobs, oversample=oversample).item()) * df
        for f in edges[1:]
    ])


def check_band_support_separation(
    band_edges,
    Tobs: float,
    stride: int,
    *,
    sep_factor: Optional[float] = None,
    oversample: int = 4,
    amp: float = 1e-30,
    context: str = "",
    enforce: bool = True,
) -> dict:
    """SUPPORT-based same-unit separation check for the GB band scheduling.

    PHYSICS RULING (user, verified premise): an FD inner product of ~0
    implies a WDM inner product of ~0, EVEN within one wavelet layer.
    The concurrency constraint for GB sub-bands is therefore
    ORTHOGONALITY (frequency separation), NOT disjoint wavelet-pixel
    support: two sources separated by ``|df| * Tobs >> 1`` have
    ``<h_i|h_j> ~ 0``, so by bilinearity their likelihood deltas add and
    their evaluations may run CONCURRENTLY in independent buffer
    components. Boundary pairs (sources near a shared band edge) are the
    one place orthogonality weakens.

    THE RULE (2026-08-15 user width ruling; supersedes both the earlier
    ``div + 1`` and layer-width-inequality formulas): separation is
    measured in units of the sources' own FD SUPPORT. A source at
    frequency ``f`` has edge half-support ``s(f) = get_N(f) / Tobs`` Hz
    (see :func:`band_support_halfwidths`). For every same-unit band pair
    ``(b, b + stride)`` -- the closest concurrent bands, with
    ``stride - 1`` closed bands between them -- the enforced inequality
    is::

        gap = edges[b + stride] - edges[b + 1]
            >= sep_factor * (s(edges[b + 1]) + s(edges[b + stride]))

    with ``sep_factor`` = ``GB_ORTHO_SEP_FACTOR`` (default 1.0 =
    edge-source supports may just touch but never overlap; 1e-9 relative
    tolerance for the exact-equality case).

    DERIVED MINIMUM STRIDE under the ``get_n`` width rule (band width =
    its own minimum ``w_b = 2 * get_N(f_hi_b) / Tobs``, maximal
    packing): for ``stride = 2`` the gap is one band,
    ``gap = w_{b+1} = 2 * s(f_hi_{b+1})``, while the overhang sum is
    ``s(f_hi_b) + s(f_hi_{b+1}) <= 2 * s(f_hi_{b+1})`` because ``get_N``
    is non-decreasing in ``f`` -- so **stride 2 always satisfies factor
    1.0** (equality exactly when ``get_N`` is flat across the middle
    band). ``stride = 3`` adds a full band of clearance
    (``gap >= overhang sum + one band width``), covering factors up to
    ~2. Grids NOT built by the width rule (legacy uniform layers,
    explicit overrides) can genuinely fail at stride 2 -- that is what
    ``min_safe_stride`` in the returned summary reports.

    Args:
        band_edges: Ascending band-edge frequencies (Hz), free-floating
            (any array-like; cupy accepted).
        Tobs: Observation time (s); ``df = 1/Tobs``.
        stride: Band-unit stride to validate (``band_units`` /
            ``GB_BAND_UNIT_STRIDE``).
        sep_factor: Safety factor on the overhang sum. ``None`` resolves
            from ``GB_ORTHO_SEP_FACTOR`` (default 1.0).
        oversample / amp: Forwarded to ``get_N`` (the production sizing
            call convention).
        context: Label prepended to the error message.
        enforce: ``True`` -> raise ``ValueError`` on failure; ``False``
            -> only report (diagnostic mode).

    Returns:
        dict with ``passes`` (bool at ``stride``), ``min_safe_stride``
        (smallest stride in [2, 16] passing, or ``None``), ``worst``
        ``(b, gap_hz, need_hz)`` for the tightest pair at ``stride``,
        and ``support_hz`` (the per-band edge half-supports).

    Raises:
        ValueError: When ``enforce`` and any same-unit pair violates the
            inequality at ``stride``.
    """
    edges = np.asarray(asnumpy(band_edges), dtype=float)
    if sep_factor is None:
        sep_factor = float(os.environ.get("GB_ORTHO_SEP_FACTOR", "1.0"))
    sep_factor = float(sep_factor)
    stride = int(stride)
    out = {
        "passes": True,
        "min_safe_stride": 2 if edges.size >= 3 else None,
        "worst": None,
        "support_hz": np.zeros(max(0, edges.size - 1)),
        "sep_factor": sep_factor,
    }
    if edges.size < 3:
        # 0 or 1 band: no two bands are ever concurrent.
        return out
    s = band_support_halfwidths(edges, Tobs, oversample=oversample, amp=amp)
    out["support_hz"] = s
    nb = edges.size - 1
    tol = 1e-9

    def _eval(k: int):
        """(passes, worst_pair) over same-unit adjacent pairs at stride k."""
        if nb <= k:
            return True, None
        b = np.arange(nb - k)
        # band b's top edge is edges[b+1] (half-support s[b]); band
        # (b+k)'s low edge is edges[b+k] (half-support s[b+k-1], the
        # get_N at that edge = the (b+k-1) band's top-edge value).
        gap = edges[b + k] - edges[b + 1]
        need = sep_factor * (s[b] + s[b + k - 1])
        margin = gap - need * (1.0 - tol)
        j = int(np.argmin(margin))
        return bool(np.all(margin >= 0.0)), (
            int(b[j]), float(gap[j]), float(need[j])
        )

    passes, worst = _eval(stride)
    out["passes"] = passes
    out["worst"] = worst
    min_safe = None
    for k in range(2, min(17, nb + 1)):
        ok, _ = _eval(k)
        if ok:
            min_safe = k
            break
    out["min_safe_stride"] = min_safe
    if enforce and not passes:
        b, gap_hz, need_hz = worst
        raise ValueError(
            f"{context + ': ' if context else ''}band-unit stride {stride} "
            f"is unsafe for this band grid: same-unit bands {b} and "
            f"{b + stride} are separated by {gap_hz:.4e} Hz but their "
            f"edge-source FD supports reach "
            f"{need_hz:.4e} Hz into the gap (sep_factor {sep_factor:.3g} x "
            f"(get_N(f_hi_low_band) + get_N(f_lo_high_band)) / Tobs). "
            f"Concurrently scheduled sources near those edges would have "
            f"overlapping supports (<h_i|h_j> not ~ 0). Minimum safe "
            f"stride for this grid: {min_safe} (rule: gap >= "
            f"GB_ORTHO_SEP_FACTOR * sum of edge half-supports; the "
            f"2*get_N width rule guarantees stride 2 at factor 1.0)."
        )
    return out


def pack_special_index(temp_inds, walker_inds, band_inds, nwalkers: int):
    """Pack ``(temp, walker, band)`` triplets into scalar special indices.

    ``special = (temp * nwalkers + walker) * 1e6 + band``. Works
    elementwise on array inputs of any matching shape.
    """
    return (temp_inds * nwalkers + walker_inds) * _SPECIAL_INDEX_BASE + band_inds


def unpack_special_index(special_band_inds, nwalkers: int) -> tuple:
    """Recover ``(temp, walker, band)`` arrays from packed special indices."""
    # input-driven array module (NOT the module-level cp): this helper runs
    # on numpy inputs during CPU-resolved runs on cupy-installed machines.
    xp = get_array_module(special_band_inds)
    temp_walker_inds = xp.floor(special_band_inds / _SPECIAL_INDEX_BASE).astype(int)
    temp_inds = temp_walker_inds // nwalkers
    walker_inds = temp_walker_inds % nwalkers
    band_inds = (special_band_inds - temp_walker_inds * _SPECIAL_INDEX_BASE).astype(int)
    return (temp_inds, walker_inds, band_inds)


def _tspan(tm, name: str):
    """Timer span or no-op when no timer is supplied.

    ``tm`` is a ``gbspecialstretch._ProposeTimer`` (or anything exposing a
    ``span(name)`` context manager) — duck-typed here so this module never
    imports from :mod:`gbspecialstretch` (acyclic import graph).
    """
    return tm.span(name) if tm is not None else nullcontext()


def return_x(x):
    """Identity helper used as a no-op replacement for :func:`copy.deepcopy`."""
    return x


def _index_asserts() -> bool:
    """Whether the O(n)-per-call index-bound asserts run (``GB_INDEX_ASSERTS=1``).

    Mirrors the :mod:`lisatools.chunked_het` gate (same env knob) but reads
    per call so tests can flip it; the read is nanoseconds against the
    asserts' cost."""
    return os.environ.get("GB_INDEX_ASSERTS", "0") == "1"


def _cell_label_deferred() -> bool:
    """Whether cell relabels accumulate instead of hitting the full table.

    ``GB_CELL_LABEL_DEFERRED`` -- **default ``"1"`` = DEFERRED** (user
    ruling 2026-08-28: sources never move between cells inside a window,
    ONLY CELLS CHANGE LABELS -- the design's own tempering invariant --
    so deferral is the native bookkeeping, not an optimization mode).
    Swaps inside a window opened by
    :meth:`BandSorter.begin_cell_label_window` compose into an O(K)
    permutation over that window's CELLS, and
    :meth:`BandSorter.flush_cell_labels` applies it to the flat source
    table in ONE pass. ``"0"`` is the escape hatch: the legacy immediate
    per-swap full-table relabel, byte-identical to the pre-2026-08-28
    behavior.

    WHY (orchestration audit 2026-08-27, candidate 2): the immediate
    relabel scans the FULL source table (``isin`` over 1e6-1e7 rows, a
    boolean-mask getitem that forces a device sync, and 3 full-table
    scatters) once per accepted tempering rung pair (~40k pairs per
    iteration) and once per vertical sweep (one per in-model repeat step)
    -- 30-150 s/row.

    Read per call so tests can flip it, exactly like :func:`_index_asserts`;
    the read is nanoseconds against a full-table pass.
    """
    return os.environ.get("GB_CELL_LABEL_DEFERRED", "1") == "1"


def _router_device_resident() -> bool:
    """Whether the shard router keeps routed tensors device-resident.

    ``GB_ROUTER_DEVICE_RESIDENT`` (default ``"1"``): params/N_vals stay on
    the caller's device and are sliced per shard there, per-shard moves are
    direct device-to-device copies inside the target ``device_context``
    (``xp.asarray`` of a foreign-device array), and outputs assemble by
    device-side scatter into a preallocated array on the caller's device —
    no host round-trips. ``"0"`` restores the legacy host-staging path
    (asnumpy + per-shard re-upload + host reassembly) — kept callable as
    cheap production insurance. Both paths are bit-identical (copies never
    change values); on numpy the two are literally the same operations.
    """
    return os.environ.get("GB_ROUTER_DEVICE_RESIDENT", "1") == "1"


def _slice_rows(xp, arr, rows):
    """``arr[rows]`` executed under ``arr``'s own device.

    Device-resident staging helper: slicing a cupy array is only legal on
    its owning device, and the router pre-slices per-shard rows on the
    CALLER's thread (workers then ``xp.asarray`` the slice inside their own
    shard context — the cross-device copy). Numpy / host arrays slice
    directly (``device_context`` no-ops)."""
    if arr is None:
        return None
    dev = getattr(getattr(arr, "device", None), "id", None)
    with device_context(xp, dev):
        return arr[rows]


class BandScheduler:
    """Staged loading of (temp, walker, band) cells through the sub-band buffer.

    The proposal loop runs one source-pick per active cell per round. This
    object owns the bookkeeping the loop needs:

    * which cells (packed special indices) currently occupy the buffer's
      ``n_subbands`` slots,
    * how many of each cell's sources have been consumed (a cell is finished
      when every one of its sources has been picked exactly once),
    * which buffer slots to swap out for pending cells when their cell
      finishes (:meth:`advance` returns the ``(inds_fill, new_specials)``
      pair that :meth:`SubBandBuffer` loading consumes).

    Cells are ordered by ascending source count so short cells retire early
    and the buffer stays densely packed with work.

    ``cell_order`` (2026-08-18) selects that ordering:

    ``"count"``
        The historical default above -- best slot packing.
    ``"band"``
        Sort by ``(band, walker, temp)`` -- temperature LAST -- so a
        vertical partner pair ``(t, w, b)`` / ``(t-1, w, b)`` lands in
        ADJACENT slots. This is what makes VERTICAL band-temperature swaps
        possible: such a pair must be resident SIMULTANEOUSLY, and under
        count-ordering that is coincidence.

        Measured partner availability (fraction of resident cells having a
        partner), simulated at 24 temps x 24 walkers:

        =========================  =======  ============  =================
        configuration              count    band,count    band,walker,temp
        =========================  =======  ============  =================
        production 77 bands, 8192    17.7%        94.2%              95.5%
        probe 4 bands, 1152 slots    47.7%        89.0%              91.8%
        probe 4 bands, 64 slots       0.0%        14.1%              89.1%
        =========================  =======  ============  =================

        Temperature-last is what carries the last row: ordering by
        ``(band, count)`` splits partners as soon as the buffer holds less
        than one full ``ntemps x nwalkers`` column.

        **The expected packing cost does not materialise.** The worry was
        that a band column mixes short and long cells, so slots would idle
        while a column's longest cell finishes. Replaying the real
        pick/``advance`` loop at production scale says otherwise -- band
        ordering needs slightly FEWER rounds and keeps slots slightly
        BUSIER:

        ==========  ==========  =================  ==================
        occupancy   rounds      slot utilisation   pair availability
        ==========  ==========  =================  ==================
        Poisson     43 -> 39    67.7% -> 74.7%     16.7% -> 70.7%
        clustered   361 -> 358   8.1% ->  8.2%     83.5% -> 85.3%
        ==========  ==========  =================  ==================

        Total picks are identical either way (every source is visited
        exactly once regardless of order), which is the correctness check
        on the simulation. Still worth watching the ``pick`` / ``advance``
        spans on the first real run -- the simulation models occupancy, not
        kernel cost.
    """

    #: Ordering modes accepted by ``cell_order``.
    CELL_ORDERS = ("count", "band")

    @property
    def xp(self):
        """Array module derived from a stored flag — never the module itself
        (raw module attributes break deepcopy/pickle of containing graphs)."""
        return cp if self._uses_cupy else np

    def __init__(self, special_band_inds, n_subbands, xp=np, cell_order="count",
                 nwalkers=None):
        # Store a flag, not the module (see the ``xp`` property).
        self._uses_cupy = (getattr(xp, "__name__", "numpy") == "cupy")
        if cell_order not in self.CELL_ORDERS:
            raise ValueError(
                f"cell_order must be one of {self.CELL_ORDERS}, "
                f"got {cell_order!r}."
            )
        self.cell_order = cell_order
        uni, counts = xp.unique(special_band_inds, return_counts=True)
        if cell_order == "band":
            if nwalkers is None:
                raise ValueError(
                    "cell_order='band' needs nwalkers to decode the packed "
                    "special index into (temp, walker, band)."
                )
            # Sort by (band, walker, TEMP) -- lexsort's LAST key is primary.
            # Temperature LAST is the point: it makes a vertical partner
            # pair, (t, w, b) and (t-1, w, b), land in ADJACENT slots, so
            # partners survive even when the buffer holds less than a full
            # band column. Ordering by (band, count) instead splits partners
            # whenever slots < column: measured 14.1% partner availability
            # against 89.1% for this ordering at 64 slots / 576-cell column.
            band = uni % _SPECIAL_INDEX_BASE
            tw = (uni // _SPECIAL_INDEX_BASE).astype(band.dtype)
            walker = tw % int(nwalkers)
            temp = tw // int(nwalkers)
            order = xp.lexsort(xp.stack((temp, walker, band)))
        else:
            order = xp.argsort(counts)
        self.cell_specials = uni[order]
        self.cell_counts = counts[order]
        self.cell_run = xp.zeros_like(self.cell_counts)
        self.n_cells = int(len(uni))
        # lookup table: special index -> cell position (cell_specials order)
        self._lookup_order = xp.argsort(self.cell_specials)
        self._specials_sorted = self.cell_specials[self._lookup_order]

        n_slots = min(int(n_subbands), self.n_cells)
        self.slot_cell = xp.arange(n_slots)
        self.slot_active = xp.ones(n_slots, dtype=bool)
        self._next_cell = n_slots

    def _cells_of(self, specials):
        """Map special indices to cell positions."""
        pos = self.xp.searchsorted(self._specials_sorted, specials, side="left")
        return self._lookup_order[pos]

    @property
    def n_slots(self) -> int:
        return len(self.slot_cell)

    @property
    def slot_specials(self):
        """Special index per buffer slot (retired slots keep their last cell)."""
        return self.cell_specials[self.slot_cell]

    @property
    def active_slot_specials(self):
        """Special indices of the slots still doing work."""
        return self.slot_specials[self.slot_active]

    def any_active(self) -> bool:
        return bool(self.xp.any(self.slot_active))

    def record_picks(self, picked_specials) -> None:
        """Count one consumed source for each cell that got a pick."""
        self.cell_run[self._cells_of(picked_specials)] += 1

    def add_counts(self, specials, deltas) -> None:
        """Live cap-transition budget adjustment (user design 2026-08-14).

        Invariant: ``cell_counts == rows picked + rows currently pickable``.
        When an accepted death frees an at-cap cell, its unpicked staged
        birth rows become pickable and are ADDED to the cell's finish
        budget; when an accepted birth re-caps a cell, its unpicked birth
        rows are SUBTRACTED again. Cells therefore finish (and retire)
        exactly when their live-pickable work is done, through any
        sequence of free/re-cap transitions -- no deadlock, no slot
        hogging. ``specials`` must be unique (at most one accept per cell
        per round guarantees this at the call site).
        """
        self.cell_counts[self._cells_of(specials)] += deltas

    def advance(self, frozen_specials=None):
        """Retire finished slots and stage pending cells into them.

        Returns ``(inds_fill, new_specials)``: the buffer slot positions to
        repack and the special indices of the cells to load there. Slots
        with no pending replacement are deactivated.

        ``frozen_specials`` (grouped RJ scheduling, 2026-08-14): cells
        holding a pending in-model source are never retired even when all
        their sources are consumed — a pending source pins its cell's
        buffer slot until the flush runs. Passing the frozen set lets the
        caller stage new cells into the OTHER finished slots so the pool
        keeps accumulating toward a full-width in-model block.
        """
        finished = self.slot_active & (
            self.cell_run[self.slot_cell] >= self.cell_counts[self.slot_cell]
        )
        if frozen_specials is not None and len(frozen_specials):
            finished &= ~self.xp.isin(self.slot_specials, frozen_specials)
        n_finished = int(finished.sum())
        if n_finished == 0:
            return self.xp.zeros(0, dtype=int), self.xp.zeros(0, dtype=int)

        n_pending = self.n_cells - self._next_cell
        n_replace = min(n_finished, n_pending)
        finished_slots = self.xp.arange(self.n_slots)[finished]

        inds_fill = finished_slots[:n_replace]
        new_cells = self._next_cell + self.xp.arange(n_replace)
        self.slot_cell[inds_fill] = new_cells
        self._next_cell += n_replace

        # slots beyond the replacements retire
        self.slot_active[finished_slots[n_replace:]] = False
        return inds_fill, self.cell_specials[new_cells]


class _ShardHolderView:
    """Single-shard holder view over one GPU split of a multi-shard ACA.

    The gbgpu band engines are single-shard by contract: they consume
    ``holder.linear_data_arr[0]`` / ``linear_psd_arr[0]``, index rows by an
    intra-buffer ``data_index``, and cache pointer-bound bindings on the
    holder. This view presents ONE split of a multi-shard
    :class:`~lisatools.analysiscontainer.AnalysisContainerArray` (a
    :class:`SubBandBuffer` or the parent residual ACA) through exactly that
    protocol:

    * ``linear_data_arr`` / ``linear_psd_arr`` are one-element lists holding
      the owning split's live buffer (zero-copy);
    * ``acs_total_entries`` is the split's row count — engine row indices
      are INTRA-shard (:class:`_RoutedBandEngine` translates);
    * ``min_freq_inds`` / ``start_freq_ind`` / ``slab_min_f`` are persistent
      per-shard stores refreshed IN PLACE from the parent
      (:meth:`refresh_row_metadata`) so the engines' pointer-binding contract
      survives cell swaps;
    * everything else (settings, df, xp, ...) delegates to the parent.

    Engine bindings (``_gb_fd_binding``) cache on this object, so a view
    must live exactly as long as its shard buffers: the router stores views
    on the holder itself (``holder._shard_holder_views``), which dies with
    the holder at proposal teardown (memory-lifecycle rule).
    """

    def __init__(self, parent, split_index: int):
        self._parent = parent
        self._split = int(split_index)
        rows = np.asarray(asnumpy(parent.gpu_splits[self._split]), dtype=int)
        self._rows = rows
        self.acs_total_entries = int(rows.shape[0])
        self.device = (
            None if parent.gpus is None else int(parent.gpus[self._split])
        )
        self.gpus = None if parent.gpus is None else [self.device]
        self.gpu_splits = [np.arange(rows.shape[0])]
        self.split_map = np.zeros(rows.shape[0], dtype=int)
        self.gpu_map = (
            np.zeros(rows.shape[0], dtype=int)
            if self.device is None
            else np.full(rows.shape[0], self.device, dtype=int)
        )
        self._min_freq_inds_view = None
        self._start_freq_ind_view = None
        self._slab_min_f_view = None
        self.refresh_row_metadata()

    @property
    def rows(self):
        """Global row ids owned by this shard (ascending)."""
        return self._rows

    @property
    def linear_data_arr(self):
        return [self._parent.linear_data_arr[self._split]]

    @property
    def linear_psd_arr(self):
        return [self._parent.linear_psd_arr[self._split]]

    @property
    def data_shaped(self):
        return [self._parent.data_shaped[self._split]]

    @property
    def psd_shaped(self):
        return [self._parent.psd_shaped[self._split]]

    @property
    def xp(self):
        return self._parent.xp

    @property
    def min_freq_inds(self):
        return self._min_freq_inds_view

    @property
    def start_freq_ind(self):
        return self._start_freq_ind_view

    @property
    def slab_min_f(self):
        """Per-slot narrow-slab layer origins SLICED to this shard's rows.

        MUST be an explicit property: ``__getattr__`` delegation would hand
        back the parent's **global-slot** array while every index the engines
        pass is **intra-shard** -- so a source in shard-1 row 0 would be
        folded against buffer slot 0's slab origin instead of its own. The
        consumers index it exactly that way: the chunked-het kernels via
        ``WDMComputationsBase._slab_kernel_args`` (``get_ll`` / ``swap_ll`` /
        ``fill_global`` / ``get_fstat_ll``) and
        ``GBSignalHetComputations.setup_in_model``
        (``slab_min_f[slots] - ind_min_f``). Mirrors the per-slot kwarg slice
        :meth:`_RoutedBandEngine.fill_template` already applies for the kwarg
        form. ``None`` when the parent carries no slab metadata (narrow slabs
        off, or the parent residual ACA).

        ``band_slab_Nf`` needs no such override: it is a scalar extent shared
        by every slab, hence shard-invariant, and keeps delegating.
        """
        return self._slab_min_f_view

    def refresh_row_metadata(self) -> None:
        """Re-slice per-row metadata from the parent.

        Updates the persistent per-shard ``min_freq_inds`` / ``slab_min_f``
        stores IN PLACE (the FD binding holds a pointer to the former)
        instead of rebinding.
        """
        xp = self._parent.xp
        starts = getattr(self._parent, "min_freq_inds", None)
        if starts is None:
            self._min_freq_inds_view = None
            vals_host = None
        else:
            vals_host = np.ascontiguousarray(
                np.asarray(asnumpy(starts))[self._rows].astype(np.int32)
            )
            with device_context(xp, self.device):
                if (
                    self._min_freq_inds_view is not None
                    and self._min_freq_inds_view.shape == vals_host.shape
                ):
                    self._min_freq_inds_view[...] = xp.asarray(vals_host)
                else:
                    self._min_freq_inds_view = xp.ascontiguousarray(
                        xp.asarray(vals_host)
                    )
        # ``start_freq_ind`` view (perf, 2026-08): on a SubBandBuffer parent
        # the generic read below resolves to AnalysisContainerArray's
        # ``start_freq_ind`` property — a per-slot Python dispatch LOOP over
        # every allocated container, run on every engine call via
        # ``_shard_views``. A SubBandBuffer already stores its per-slot
        # window starts (``start_freq_inds`` -> the capacity-sized
        # ``min_freq_inds`` store, sliced to this shard's rows in
        # ``vals_host`` above), so reuse that and skip the ACA loop. The
        # only consumer of this view (``GBFDComputations._holder_starts``)
        # prefers ``min_freq_inds`` whenever it is present — as it always is
        # on a SubBandBuffer — so this fallback is never read on that path.
        if vals_host is not None and getattr(
                self._parent, "start_freq_inds", None) is not None:
            self._start_freq_ind_view = vals_host
        else:
            sfi = getattr(self._parent, "start_freq_ind", None)
            if sfi is None:
                self._start_freq_ind_view = None
            else:
                arr = np.asarray(asnumpy(sfi))
                self._start_freq_ind_view = arr[self._rows] if arr.ndim else arr
        # Narrow per-band slab origins: one value PER BUFFER SLOT, so the
        # shard's view must carry its own rows' values in intra-shard order
        # (see the ``slab_min_f`` property). Refreshed in place like
        # ``min_freq_inds`` so a cell swap on the parent reaches every view.
        slab = getattr(self._parent, "slab_min_f", None)
        if slab is None:
            self._slab_min_f_view = None
        else:
            slab_host = np.ascontiguousarray(
                np.asarray(asnumpy(slab))[self._rows].astype(np.int32)
            )
            with device_context(xp, self.device):
                if (
                    self._slab_min_f_view is not None
                    and self._slab_min_f_view.shape == slab_host.shape
                ):
                    self._slab_min_f_view[...] = xp.asarray(slab_host)
                else:
                    self._slab_min_f_view = xp.ascontiguousarray(
                        xp.asarray(slab_host)
                    )

    def __len__(self) -> int:
        # The engines read ``len(holder)`` as the shard's row (cell) count
        # -> ``num_data``/``num_noise`` for the flat single-shard buffer.
        # MUST be an explicit dunder: ``len()`` resolves ``__len__`` on the
        # TYPE, bypassing ``__getattr__`` delegation, so a parent-forwarded
        # ``__len__`` is never seen. Mirrors AnalysisContainerArray.__len__
        # (== the number of containers on this split).
        return int(self.acs_total_entries)

    def __getattr__(self, name):
        # Guard dunder/underscore probing (deepcopy/pickle safety rule);
        # delegate the public long tail (settings, df, nchannels, ...).
        # NOTE: implicitly-invoked dunders (len(), iter(), ...) resolve on
        # the type and never reach here -- define each one explicitly above.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._parent, name)


class _FStatRefRowHolder:
    """One-slab wdm_holder replica feeding one device's sig-het F-stat lane.

    ``GBSignalHetComputations.setup_fstat_references`` consumes exactly
    ``linear_data_arr[0]`` / ``linear_psd_arr[0]`` (full-band slab layouts)
    picked by SCALAR ``data_index`` / ``noise_index``, and
    ``get_fstat_ll_wdm`` never touches the holder at all -- so a scoring
    lane needs only the ONE reference walker's residual + inverse-PSD rows,
    copied onto its device once at adapter build (host-routed, no P2P) and
    presented as a single-slab holder scored with ``data_index=0``.
    Private to :meth:`_RoutedBandEngine._sighet_fstat_multidevice`;
    ``device`` / ``gpus`` / ``__len__`` mirror :class:`_ShardHolderView` so
    the router's replica plumbing (``_comp_for``) accepts it as a view.
    Lives only in the returned ``call_fstat`` closure -- it dies with the
    fit, and the copies never reach the settings tree.
    """

    def __init__(self, parent, device, data_row, psd_row):
        self._parent = parent
        self.device = None if device is None else int(device)
        self.gpus = None if self.device is None else [self.device]
        self.acs_total_entries = 1
        self.linear_data_arr = [data_row]
        self.linear_psd_arr = [psd_row]

    @property
    def xp(self):
        return self._parent.xp

    def __len__(self) -> int:
        # The comps read ``len(holder)`` as the slab count (explicit dunder:
        # len() resolves on the type, bypassing ``__getattr__``).
        return 1

    def __getattr__(self, name):
        # Same delegation discipline as _ShardHolderView: guard underscore
        # probing, delegate the public long tail to the parent ACA.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._parent, name)


class _RoutedBandEngine:
    """Multi-shard router in front of a single-shard band likelihood engine.

    Wraps a :func:`gbgpu.gb_likelihood.make_band_likelihood_engine` product.
    Single-shard holders pass straight through (no overhead). For
    multi-shard holders each call's rows are partitioned by owning split
    (``holder.split_map``), run per shard inside the owning device context
    against a persistent :class:`_ShardHolderView`, and the outputs are
    reassembled full-length on the caller's device. Cross-shard movement is
    host-routed (no P2P), matching the ACA conventions. Per-launch
    host->device upload of wrapper structs (LISA Analysis Tools-wide
    convention) keeps kernel config device-local under each context.

    The GB comps a shard's kernels read (chunk geometry, WDM window,
    ``OrbitsWrap`` / ``TDIConfigWrap`` pointer fields, and under
    ``GB_SIGHET_INMODEL`` the whole heterodyne reference stash) are allocated
    once, on the device current at variant-build time. A shard launching on a
    different device would dereference them across the PCIe link -- a silent
    peer-access tax with P2P, an illegal access without it -- and, for
    sig-het, would share ONE reference stash, ONE slot->reference map and ONE
    ``_in_model`` flag between shards whose slot ids both start at zero. Both
    are closed by per-device replicas: ``engine_factory`` (supplied by the
    two construction sites) rebuilds the whole engine around device-local
    comps for any shard whose device differs from the prototype's, and the
    raw-comp class methods resolve the same replica through
    :meth:`_comp_for`. The prototype's own device reuses the existing engine
    object, so single-shard / primary-shard behaviour is unchanged and
    allocates nothing.
    """

    #: fill_template kwargs holding one value PER BUFFER SLOT — sliced to the
    #: shard's rows so intra-shard indexing stays aligned.
    _PER_SLOT_KWARGS = ("slab_min_f",)

    def __init__(self, engine, engine_factory=None):
        self._engine = engine
        # device -> engine replica, populated lazily on the first multi-shard
        # call that lands on a non-prototype device. The prototype's device
        # maps to ``engine`` ITSELF (never a copy), so ``len(gpus) <= 1``
        # returns the same object and allocates nothing.
        self._engine_factory = engine_factory
        self._engine_by_device = {}

    @property
    def wrapped_engine(self):
        """The underlying single-shard engine."""
        return self._engine

    @property
    def device_engines(self) -> dict:
        """``{device: engine replica}`` built so far (diagnostics/tests)."""
        return self._engine_by_device

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._engine, name)

    # ---------------- per-device comp / engine replicas ----------------

    @staticmethod
    def _comp_build_device(comp):
        """The CUDA device a GB comp's buffers were allocated on, or None.

        ``_build_device`` is recorded by ``WDMComputationsBase.__init__`` and
        ``GBFDComputations.__init__``. The sig-het wrapper records none of its
        own -- ``for_band_engine`` runs in the same device context as its
        chunked delegate -- so it reports the delegate's. The final fallback
        reads residency straight off a known device buffer, which keeps the
        answer meaningful for a comp built before the recording existed.
        """
        dev = getattr(comp, "_build_device", None)
        if dev is None:
            dev = getattr(getattr(comp, "chunked", None), "_build_device", None)
        if dev is None:
            dev = getattr(getattr(comp, "wdm_window", None), "device", None)
            dev = getattr(dev, "id", None)
        return None if dev is None else int(dev)

    @classmethod
    def _engine_comp(cls, engine):
        """The GB comp an engine scores through (WDM or FD), or None."""
        comp = getattr(engine, "gb_comps", None)
        if comp is None:
            comp = getattr(engine, "gb_fd_comp", None)
        return comp

    @classmethod
    def _primary_device(cls, comp, holder):
        """Device whose shard reuses ``comp`` unchanged.

        The comp's OWN build device when it recorded one -- not blindly
        ``gpus[0]``: a comp constructed before the run pinned its main device
        lives on device 0 even when ``gpus=[2, 3]``, and keying on the
        recorded value replicates for every shard (correct) instead of
        handing device-0 pointers to the ``gpus[0]`` shard (wrong).
        """
        dev = None if comp is None else cls._comp_build_device(comp)
        if dev is not None:
            return dev
        gpus = getattr(holder, "gpus", None)
        return None if not gpus else int(gpus[0])

    @classmethod
    def _assert_comp_device(cls, comp, view):
        """Fail loudly when a shard is about to launch on foreign buffers.

        Cheap permanent guard: the moment a new comp-level device buffer is
        added without a matching replica path, this turns "mysteriously slow,
        or an illegal access on a non-P2P node" into a message that names the
        fix. A comp that records nothing (CPU, or an unrecognised comp type)
        is skipped rather than guessed at.
        """
        dev = getattr(view, "device", None)
        if comp is None or dev is None:
            return
        build_dev = cls._comp_build_device(comp)
        if build_dev is not None and build_dev != int(dev):
            raise RuntimeError(
                f"GB comp {type(comp).__name__} holds buffers on device "
                f"{build_dev} but this shard launches on device {dev}: the "
                "kernel would read across devices (a silent P2P tax, or an "
                "illegal access on a node without peer access). A per-device "
                "comp replica is needed -- see _device_local_gb_comp in "
                "lisatools.globalfit.stock.erebor.source_runtime."
            )

    @classmethod
    def _comp_for(cls, comp, holder, view):
        """The device-local replica of ``comp`` for this shard's device."""
        dev = getattr(view, "device", None)
        if comp is None or dev is None:
            return comp
        from ..stock.erebor.source_runtime import _device_local_gb_comp

        out = _device_local_gb_comp(
            comp, holder.xp, int(dev), cls._primary_device(comp, holder)
        )
        cls._assert_comp_device(out, view)
        return out

    def _engine_for(self, holder, view):
        """The likelihood engine this shard must run on.

        The prototype's device (and any holder/engine without the metadata to
        do better) gets ``self._engine`` itself. Every other device gets one
        cached replica built through ``engine_factory`` -- which rebuilds the
        engine around ``_device_local_gb_comp`` comps -- so the shard's
        kernels, its coefficient stash and its ``_in_model`` state are all
        its own.
        """
        dev = getattr(view, "device", None)
        if dev is None or self._engine_factory is None:
            return self._engine
        primary = self._primary_device(self._engine_comp(self._engine), holder)
        if primary is not None and int(dev) == int(primary):
            return self._engine
        engine = self._engine_by_device.get(int(dev))
        if engine is None:
            engine = self._engine_factory(int(dev), primary)
            self._engine_by_device[int(dev)] = engine
        self._assert_comp_device(self._engine_comp(engine), view)
        return engine

    # ---------------- shard bookkeeping ----------------

    @staticmethod
    def _is_multi(holder) -> bool:
        return len(holder.linear_data_arr) > 1

    @staticmethod
    def _shard_views(holder):
        views = getattr(holder, "_shard_holder_views", None)
        n = len(holder.linear_data_arr)
        if views is None or len(views) != n:
            views = [_ShardHolderView(holder, s) for s in range(n)]
            holder._shard_holder_views = views
        else:
            for v in views:
                v.refresh_row_metadata()
        return views

    @staticmethod
    def _partition(holder, data_index, noise_index=None):
        """Per-shard ``(positions, intra_data, intra_noise)`` partition.

        ``positions`` index into the call's row batch; ``intra_*`` are the
        corresponding intra-shard buffer rows. ``data_index`` and
        ``noise_index`` rows must be co-located on the same shard.
        """
        idx = np.asarray(asnumpy(data_index), dtype=int)
        # Static lookup, cached on the holder (perf, 2026-08): the
        # walker/slot -> (shard, intra) mapping derives from the static
        # gpu_splits, so rebuilding it O(capacity) per routed call was pure
        # overhead. shard_lookup_maps caches it on the ACA once.
        split_map, intra = shard_lookup_maps(holder)
        nidx = None
        if noise_index is not None:
            nidx = np.asarray(asnumpy(noise_index), dtype=int)
            if not np.array_equal(split_map[idx], split_map[nidx]):
                raise ValueError(
                    "data_index and noise_index rows must live on the same "
                    "shard (cross-shard noise rows are unsupported)."
                )
        parts = []
        for s in range(len(holder.linear_data_arr)):
            pos = np.where(split_map[idx] == s)[0]
            parts.append((
                pos,
                intra[idx[pos]],
                None if nidx is None else intra[nidx[pos]],
            ))
        return parts

    @staticmethod
    def _assemble(num, pieces, default, xp):
        """Host-assemble per-shard outputs into one xp array (row order).

        ``pieces`` is ``[(positions, host_values_or_None), ...]``. Returns
        None when every shard produced None (e.g. ``phase_angle`` without
        phase maximisation).
        """
        first = next(
            (np.asarray(v) for _, v in pieces if v is not None), None
        )
        if first is None:
            return None
        out = np.full(
            (num,) + tuple(first.shape[1:]), default, dtype=first.dtype
        )
        for pos, vals in pieces:
            if vals is None or len(pos) == 0:
                continue
            out[pos] = np.asarray(vals)
        return xp.asarray(out)

    @staticmethod
    def _assemble_device(num, pieces, default, xp, device):
        """Device-side scatter-assemble (GB_ROUTER_DEVICE_RESIDENT path).

        ``pieces`` is ``[(positions, device_values_or_None), ...]`` with each
        value array living on its shard's device. The output is preallocated
        on the CALLER's device (``device``) and each shard's rows are
        scattered in via ``xp.asarray`` — a direct device-to-device copy
        inside the target context, no host round-trip. Returns None when
        every shard produced None (mirrors :meth:`_assemble`).
        """
        first = next((v for _, v in pieces if v is not None), None)
        if first is None:
            return None
        with device_context(xp, device):
            out = xp.full(
                (num,) + tuple(first.shape[1:]), default, dtype=first.dtype
            )
            for pos, vals in pieces:
                if vals is None or len(pos) == 0:
                    continue
                out[pos] = xp.asarray(vals)
        return out

    @classmethod
    def _make_assembler(cls, num, xp, dev_resident):
        """(take, assemble) pair for one routed call.

        ``take`` post-processes a per-shard output inside the shard's own
        device context (host pull on the legacy path, identity when
        device-resident); ``assemble`` reassembles the collected pieces
        full-length on the caller's device. Factored so every routed leg
        stages/assembles identically."""
        if dev_resident:
            caller_dev = current_device(xp)

            def assemble(pieces, default):
                return cls._assemble_device(num, pieces, default, xp, caller_dev)

            return (lambda v: v), assemble

        def assemble(pieces, default):
            return cls._assemble(num, pieces, default, xp)

        return asnumpy, assemble

    @staticmethod
    def _dispatch_shards(holder, items, worker, state_ids=None):
        """Run ``worker(*item)`` once per populated-shard work item.

        THREADED by default (2026-08-13, user ruling): the 2-GPU
        incremental-ll-drift investigation that justified the serial safety
        net CLOSED 2026-08-12 (three root causes fixed, comp replicas
        GPU-verified bit-identical), so shards now run concurrently -- one
        host thread per shard with work, each entering its own device
        context (cupy's current device is thread-local; kernel launches
        release the GIL), mirroring
        ``AnalysisContainerArray._run_per_split``. Callers pre-resolve
        engines/comp replicas and pre-size one result slot per shard BEFORE
        dispatch, so the threaded path shares no mutable state beyond
        disjoint slot writes and reassembly stays deterministic (the
        serial and threaded paths write identical slots).

        ``state_ids``: one hashable per item naming the stateful object the
        item runs on (engine / comp replica). Items sharing one (missing
        engine_factory, two shards on one device, CPU fakes) would race
        that object's output attrs, so any duplicate forces serial.
        ``holder.thread_pool`` is only touched on the threaded branch (the
        real ACA property allocates the pool on first access).
        ``GB_ROUTER_THREADED=0`` restores serial launch-sync-launch
        dispatch (the drift checks / [GB_CELL_LL] reconciles are the
        regression alarms if concurrency ever misbehaves).
        """
        if (len(items) > 1
                and os.environ.get("GB_ROUTER_THREADED", "1") == "1"
                and (state_ids is None or len(set(state_ids)) == len(items))
                and getattr(holder, "thread_pool", None) is not None):
            futures = [holder.thread_pool.submit(worker, *it) for it in items]
            for f in futures:
                f.result()  # re-raise worker exceptions in caller
        else:
            for it in items:
                worker(*it)

    def _mirror_engine_outputs(self):
        """Refresh routed-output attrs from the wrapped engine after a
        passthrough call so stale routed values never shadow them."""
        for name in ("d_h_out", "h_h_out", "phase_angle", "kept_out",
                     "non_marg_d_h"):
            if hasattr(self._engine, name):
                setattr(self, name, getattr(self._engine, name))

    # ---------------- routed engine protocol ----------------

    def fill_template(self, holder, params_phys, params_index, N_vals, *,
                      factor, waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            return self._engine.fill_template(
                holder, params_phys, params_index, N_vals,
                factor=factor, waveform_kwargs=waveform_kwargs, **kwargs)
        xp = holder.xp
        # Speed-diagnosis spans (previously invisible in [GB_TIMING]): same
        # route_host_stage / route_dispatch names as get_ll.
        _rtm = getattr(holder, "_prop_timer", None)
        if _rtm is not None:
            _rtm.count("route_fill_calls")
        dev_res = _router_device_resident()
        with _tspan(_rtm, "route_host_stage"):
            views = self._shard_views(holder)
            parts = self._partition(holder, params_index)
            params_src = params_phys if dev_res else asnumpy(params_phys)
            N_src = (N_vals if dev_res
                     else (None if N_vals is None else asnumpy(N_vals)))
            slot_src = {
                k: (kwargs[k] if dev_res else np.asarray(asnumpy(kwargs[k])))
                for k in self._PER_SLOT_KWARGS
                if kwargs.get(k) is not None
            }
            # Per-shard slices staged on the caller's thread/device (see
            # _slice_rows); workers move them with one xp.asarray each.
            items = [
                (view, self._engine_for(holder, view), pos, intra,
                 _slice_rows(xp, params_src, pos),
                 _slice_rows(xp, N_src, pos),
                 {k: _slice_rows(xp, v, view.rows)
                  for k, v in slot_src.items()})
                for view, (pos, intra, _) in zip(views, parts)
                if pos.shape[0]
            ]

        def _shard(view, engine, pos, intra, p_part, n_part, slot_parts):
            kw_s = dict(kwargs)
            with device_context(xp, view.device):
                for k, part in slot_parts.items():
                    kw_s[k] = xp.asarray(part)
                engine.fill_template(
                    view, xp.asarray(p_part), intra,
                    None if n_part is None else xp.asarray(n_part),
                    factor=factor, waveform_kwargs=waveform_kwargs, **kw_s)

        with _tspan(_rtm, "route_dispatch"):
            self._dispatch_shards(holder, items, _shard,
                                  state_ids=[id(it[1]) for it in items])
        # Drift-hunt debug fence (see gbspecialstretch run_proposal): the
        # fills above are enqueued on each shard's own device stream; a
        # caller that immediately reads the filled shard from another
        # device (peer access) is not ordered against them.
        if (os.environ.get("GB_MULTIGPU_SYNC_DEBUG", "0") == "1"
                and getattr(xp, "cuda", None) is not None):
            for view in views:
                if view.device is not None:
                    with device_context(xp, view.device):
                        xp.cuda.runtime.deviceSynchronize()

    def get_ll(self, holder, params_phys, *, data_index, noise_index,
               N_vals, phase_maximize=False, waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            _rtm0 = getattr(holder, "_prop_timer", None)
            if _rtm0 is not None:
                _rtm0.count("route_single_calls")
            with _tspan(_rtm0, "route_single_engine"):
                out = self._engine.get_ll(
                    holder, params_phys, data_index=data_index,
                    noise_index=noise_index, N_vals=N_vals,
                    phase_maximize=phase_maximize,
                    waveform_kwargs=waveform_kwargs, **kwargs)
            self._mirror_engine_outputs()
            return out
        xp = holder.xp
        # Speed-diagnosis spans (user directive 2026-08-15). PRIME SUSPECT
        # for the bench-vs-production gap: this multi-shard path host-stages
        # EVERY call — asnumpy(params) is a full device sync, each shard
        # re-uploads its slice, and each shard's FIVE outputs come back via
        # asnumpy (more syncs) before _assemble re-uploads. The benches are
        # single-device and never pay any of it. route_host_stage /
        # route_dispatch / route_assemble attribute the three phases.
        _rtm = getattr(holder, "_prop_timer", None)
        if _rtm is not None:
            _rtm.count("route_multi_calls")
        dev_res = _router_device_resident()
        with _tspan(_rtm, "route_host_stage"):
            views = self._shard_views(holder)
            parts = self._partition(holder, data_index, noise_index)
            num = int(params_phys.shape[0])
            take, assemble = self._make_assembler(num, xp, dev_res)
            # Device-resident (default): params/N_vals never leave the
            # caller's device — per-shard row slices are staged here on the
            # caller's thread and each worker moves its slice with ONE
            # xp.asarray (a direct device-to-device copy in its own
            # context). Legacy (GB_ROUTER_DEVICE_RESIDENT=0): asnumpy
            # host-stage + per-shard re-upload, unchanged.
            params_src = params_phys if dev_res else asnumpy(params_phys)
            N_src = (N_vals if dev_res
                     else (None if N_vals is None else asnumpy(N_vals)))
            items = [
                (si, view, self._engine_for(holder, view), pos, intra,
                 intra_noise,
                 _slice_rows(xp, params_src, pos),
                 _slice_rows(xp, N_src, pos))
                for si, (view, (pos, intra, intra_noise))
                in enumerate(zip(views, parts))
                if pos.shape[0]
            ]
        # One pre-sized slot per shard (threaded dispatch writes disjoint
        # slots; serial dispatch fills the same slots in the same order).
        slots = {name: [None] * len(views)
                 for name in ("ll", "dh", "hh", "ang", "kept", "nm")}

        def _shard(si, view, engine, pos, intra, intra_noise, p_part, n_part):
            with device_context(xp, view.device):
                ll_s = engine.get_ll(
                    view, xp.asarray(p_part),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if n_part is None else xp.asarray(n_part),
                    phase_maximize=phase_maximize,
                    waveform_kwargs=waveform_kwargs, **kwargs)
                slots["ll"][si] = (pos, take(ll_s))
                slots["dh"][si] = (pos, take(engine.d_h_out))
                slots["hh"][si] = (pos, take(engine.h_h_out))
                ang = getattr(engine, "phase_angle", None)
                slots["ang"][si] = (pos,
                                    None if ang is None else take(ang))
                kept = getattr(engine, "kept_out", None)
                slots["kept"][si] = (pos,
                                     None if kept is None else take(kept))
                # Phase-max bookkeeping: the un-maximized <d|h> the replace
                # move reads for the old side's ACTUAL-phase delta. Gathered
                # only when this call maximized -- an engine attr surviving
                # from an earlier call must never be scattered as if fresh.
                nm = (getattr(engine, "non_marg_d_h", None)
                      if phase_maximize else None)
                slots["nm"][si] = (pos, None if nm is None else take(nm))

        with _tspan(_rtm, "route_dispatch"):
            self._dispatch_shards(holder, items, _shard,
                                  state_ids=[id(it[2]) for it in items])
        with _tspan(_rtm, "route_assemble"):
            ll_p, dh_p, hh_p, ang_p, kept_p, nm_p = (
                [p for p in slots[name] if p is not None]
                for name in ("ll", "dh", "hh", "ang", "kept", "nm"))
            ll = assemble(ll_p, -1e300)
            if ll is None:
                ll = xp.full(num, -1e300)
            self.d_h_out = assemble(dh_p, 0.0)
            self.h_h_out = assemble(hh_p, 0.0)
            self.phase_angle = assemble(ang_p, 0.0)
            # None when unmaximized / no shard produced it (consumers guard).
            self.non_marg_d_h = assemble(nm_p, 0.0)
            kept_arr = assemble(kept_p, False)
            self.kept_out = (
                xp.ones(num, dtype=bool) if kept_arr is None else kept_arr
            )
        return ll

    def get_swap_ll(self, holder, params_remove_phys, params_add_phys, *,
                    data_index, noise_index, N_vals, phase_maximize=False,
                    waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            return self._engine.get_swap_ll(
                holder, params_remove_phys, params_add_phys,
                data_index=data_index, noise_index=noise_index,
                N_vals=N_vals, phase_maximize=phase_maximize,
                waveform_kwargs=waveform_kwargs, **kwargs)
        from gbgpu.gb_likelihood import SwapLLResult

        xp = holder.xp
        # Speed-diagnosis spans (previously invisible in [GB_TIMING]):
        # this leg made ~24 D2H + 13 H2D transfers per call host-staged.
        _rtm = getattr(holder, "_prop_timer", None)
        if _rtm is not None:
            _rtm.count("route_swap_calls")
        dev_res = _router_device_resident()
        with _tspan(_rtm, "route_host_stage"):
            views = self._shard_views(holder)
            parts = self._partition(holder, data_index, noise_index)
            num = int(params_add_phys.shape[0])
            take, assemble = self._make_assembler(num, xp, dev_res)
            rem_src = params_remove_phys if dev_res else asnumpy(params_remove_phys)
            add_src = params_add_phys if dev_res else asnumpy(params_add_phys)
            N_src = (N_vals if dev_res
                     else (None if N_vals is None else asnumpy(N_vals)))
            items = [
                (si, view, self._engine_for(holder, view), pos, intra,
                 intra_noise,
                 _slice_rows(xp, rem_src, pos),
                 _slice_rows(xp, add_src, pos),
                 _slice_rows(xp, N_src, pos))
                for si, (view, (pos, intra, intra_noise))
                in enumerate(zip(views, parts))
                if pos.shape[0]
            ]
        fields = ("ll_diff", "d_h_add", "d_h_remove", "hh_add",
                  "hh_remove", "hh_cross", "opt_snr_add", "phase_angle",
                  "kept")
        # Pre-sized per-shard slots (see get_ll).
        pieces = {f: [None] * len(views) for f in fields}

        def _shard(si, view, engine, pos, intra, intra_noise, r_part,
                   a_part, n_part):
            with device_context(xp, view.device):
                res = engine.get_swap_ll(
                    view, xp.asarray(r_part), xp.asarray(a_part),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if n_part is None else xp.asarray(n_part),
                    phase_maximize=phase_maximize,
                    waveform_kwargs=waveform_kwargs, **kwargs)
                for f in fields:
                    v = getattr(res, f)
                    pieces[f][si] = (pos, None if v is None else take(v))

        with _tspan(_rtm, "route_dispatch"):
            self._dispatch_shards(holder, items, _shard,
                                  state_ids=[id(it[2]) for it in items])
        with _tspan(_rtm, "route_assemble"):
            defaults = dict(ll_diff=-1e300, opt_snr_add=0.0, kept=False)
            out = {}
            for f in fields:
                out[f] = assemble(
                    [p for p in pieces[f] if p is not None],
                    defaults.get(f, 0.0))
        if out["ll_diff"] is None:
            out["ll_diff"] = xp.full(num, -1e300)
        if out["opt_snr_add"] is None:
            out["opt_snr_add"] = xp.zeros(num)
        if out["kept"] is None:
            out["kept"] = xp.zeros(num, dtype=bool)
        return SwapLLResult(**out)

    def setup_in_model(self, holder, params_phys, data_index, N_vals=None):
        """Route the per-shard in-model reference build (sig-het).

        A truthy return means the comp built heterodyne references: a
        coefficient stash, a slot->reference map and an ``_in_model`` flag,
        all held on the comp and all indexed by INTRA-shard slot ids. Two
        shards therefore need two comps -- their slot ids both start at zero,
        so a shared comp would have the second shard's build silently PATCH
        the first shard's references (the ``_in_model`` flag makes the second
        call take the mid-block patch branch), and every subsequent
        ``get_ll`` would resolve references through the wrong map. With
        ``engine_factory`` supplied each shard resolves to its own engine and
        its own comp, so each takes the fresh-build branch against its own
        residual slabs. Without one, the collision is refused loudly rather
        than computed wrongly.
        """
        if not self._is_multi(holder):
            return self._engine.setup_in_model(
                holder, params_phys, data_index, N_vals=N_vals)
        xp = holder.xp
        # Speed-diagnosis spans (previously invisible in [GB_TIMING]).
        _rtm = getattr(holder, "_prop_timer", None)
        if _rtm is not None:
            _rtm.count("route_inmodel_calls")
        dev_res = _router_device_resident()
        with _tspan(_rtm, "route_host_stage"):
            views = self._shard_views(holder)
            parts = self._partition(holder, data_index)
            params_src = params_phys if dev_res else asnumpy(params_phys)
            N_src = (N_vals if dev_res
                     else (None if N_vals is None else asnumpy(N_vals)))
        built_on = set()

        def _shard(view, engine, pos, intra, p_part, n_part):
            with device_context(xp, view.device):
                ret = engine.setup_in_model(
                    view, xp.asarray(p_part), intra,
                    N_vals=None if n_part is None else xp.asarray(n_part))
            if not ret:
                return
            # Collision only exists when two shards share ONE engine, and
            # duplicate state_ids force _dispatch_shards serial -- so this
            # check never races (threaded items all run distinct engines).
            if id(engine) in built_on:
                self.clear_in_model()
                raise NotImplementedError(
                    "sig-het in-model references are per-comp state, but two "
                    "shards resolved to the SAME likelihood engine: the "
                    "second shard's build would patch the first's references "
                    "(intra-shard slot ids collide by construction). Build "
                    "the router with engine_factory= so every shard device "
                    "gets its own comp replica, or run the sig-het in-model "
                    "path on a single GPU."
                )
            built_on.add(id(engine))

        with _tspan(_rtm, "route_host_stage"):
            items = [
                (view, self._engine_for(holder, view), pos, intra,
                 _slice_rows(xp, params_src, pos),
                 _slice_rows(xp, N_src, pos))
                for view, (pos, intra, _) in zip(views, parts)
                if pos.shape[0]
            ]
        with _tspan(_rtm, "route_dispatch"):
            self._dispatch_shards(holder, items, _shard,
                                  state_ids=[id(it[1]) for it in items])
        # ``built_on`` is populated by _shard, and _dispatch_shards has joined
        # every worker by the time it returns, so this read is not racing.
        #
        # This used to ``return None``, which swallowed every shard's truthy
        # return: on >= 2 GPUs the caller's ``sighet_active`` was ALWAYS
        # False, silently disabling the anchor check, the end-of-block
        # exact-vs-sig-het likelihood audit, the mid-block reference refresh
        # and the sig-het trust-region gate -- i.e. every diagnostic that
        # would have measured whether the in-model references were sound, on
        # exactly the multi-GPU configuration production runs on.
        return bool(built_on)

    def clear_in_model(self):
        """Clear the in-model reference on EVERY per-device engine.

        Missed fan-out is a silent bug, not an error: a replica that keeps
        ``_in_model`` set makes the next block's first ``setup_in_model`` on
        that device take the mid-block patch branch against a stale slot map.
        """
        for engine in self._engine_by_device.values():
            engine.clear_in_model()
        return self._engine.clear_in_model()

    def _route_matrix(self, method_name, holder, params_phys, *, data_index,
                      noise_index, N_vals, **kwargs):
        """Shared row-wise routing for matrix-valued outputs (grad/hessian)."""
        if not self._is_multi(holder):
            return getattr(self._engine, method_name)(
                holder, params_phys, data_index=data_index,
                noise_index=noise_index, N_vals=N_vals, **kwargs)
        xp = holder.xp
        _rtm = getattr(holder, "_prop_timer", None)
        dev_res = _router_device_resident()
        with _tspan(_rtm, "route_host_stage"):
            views = self._shard_views(holder)
            parts = self._partition(holder, data_index, noise_index)
            num = int(params_phys.shape[0])
            take, assemble = self._make_assembler(num, xp, dev_res)
            params_src = params_phys if dev_res else asnumpy(params_phys)
            N_src = (N_vals if dev_res
                     else (None if N_vals is None else asnumpy(N_vals)))
            items = [
                (si, view, self._engine_for(holder, view), pos, intra,
                 intra_noise,
                 _slice_rows(xp, params_src, pos),
                 _slice_rows(xp, N_src, pos))
                for si, (view, (pos, intra, intra_noise))
                in enumerate(zip(views, parts))
                if pos.shape[0]
            ]
        pieces = [None] * len(views)

        def _shard(si, view, engine, pos, intra, intra_noise, p_part, n_part):
            method = getattr(engine, method_name)
            with device_context(xp, view.device):
                out_s = method(
                    view, xp.asarray(p_part),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if n_part is None else xp.asarray(n_part),
                    **kwargs)
                pieces[si] = (pos, take(out_s))

        with _tspan(_rtm, "route_dispatch"):
            self._dispatch_shards(holder, items, _shard,
                                  state_ids=[id(it[2]) for it in items])
        with _tspan(_rtm, "route_assemble"):
            return assemble([p for p in pieces if p is not None], 0.0)

    def get_ll_grad(self, holder, params_phys, *, data_index, noise_index,
                    N_vals, **kwargs):
        return self._route_matrix(
            "get_ll_grad", holder, params_phys, data_index=data_index,
            noise_index=noise_index, N_vals=N_vals, **kwargs)

    def hessian(self, holder, params_phys, *, data_index, noise_index,
                N_vals, **kwargs):
        return self._route_matrix(
            "hessian", holder, params_phys, data_index=data_index,
            noise_index=noise_index, N_vals=N_vals, **kwargs)

    @classmethod
    def route_information_matrix(cls, comp, holder, params_phys, *, inds,
                                noise_index, data_index=None,
                                slot_holder=None, **swap_kwargs):
        """Route ``comp.information_matrix`` per shard.

        The Fisher/information matrix is computed on the RAW GB comp
        (``gb_wdm_comp`` / ``gb_fd_comp``) for the proposal Cholesky, not on
        the wrapped likelihood engine -- so it can't go through the instance
        router and needs its own entry point. Each binary's matrix depends
        only on its walker's PSD (``noise_index``; the data slab is
        irrelevant, per ``information_matrix``), so partition binaries by the
        owning shard of their walker, compute per shard against a persistent
        :class:`_ShardHolderView` inside the owning device context, and
        reassemble the ``(num_bin, nd, nd)`` stack on the caller's device.
        Single-shard holders pass straight through.

        Each shard runs against its own device-local comp replica
        (:meth:`_comp_for`), so the kernel never dereferences another
        device's chunk geometry / window / wrap pointers.

        SLOT-SPACE routing (2026-08-14 sig-het infomat audit): when
        ``data_index`` is supplied it means per-source BUFFER SLOT -- the
        sig-het in-model fast leg scores through ``get_ll_wdm`` against the
        reference stash ``setup_in_model`` built, whose per-device
        ``_slot_to_ref`` maps are keyed by the SAME partition
        ``setup_in_model`` used: the BUFFER holder's slot shards (bands
        parity-round-robin), NOT the parent ACA's contiguous walker shards.
        On that leg the walker's PSD rows are irrelevant (the invC is baked
        into the reference stash), so pass ``slot_holder=`` (the
        :class:`SubBandBuffer`) and the route partitions by the buffer's
        ``split_map``, remaps global slots to intra-shard rows, and
        dispatches each shard to the module-cached comp replica for its
        device -- exactly the replica whose stash holds those rows'
        references (``setup_in_model`` resolved its engines through the same
        ``_device_local_gb_comp`` cache). Without ``slot_holder`` a
        multi-shard call cannot route slots coherently: ``data_index`` is
        DROPPED with a warning and the comp takes its validated chunked
        (walker-routed) branch instead -- a graceful degrade, where
        forwarding global slots would hit the wrong replica in the wrong
        slot space (the GBGPU-side guards raise on it). The chunked/raw-comp
        leg (``data_index=None``) keeps the walker partition unchanged.
        """
        if (data_index is not None and slot_holder is not None
                and cls._is_multi(slot_holder)):
            # Route by the BUFFER's slot shards whenever the buffer itself
            # is sharded -- the slot spaces are set by the buffer, not the
            # parent ACA (in practice both shard over the same gpus list).
            return cls._route_infomat_by_slots(
                comp, slot_holder, params_phys, inds=inds,
                data_index=data_index, **swap_kwargs)
        if data_index is not None and cls._is_multi(holder):
            # MULTI-SHARD GATE (2026-08-14 sig-het infomat audit): global
            # buffer-slot ids cannot be forwarded through the WALKER-shard
            # partition below -- wrong replica (the stash lives on the
            # buffer-slot shard's comp) and wrong slot space (per-device
            # maps are keyed intra-shard). Drop them so the sig-het wrapper
            # takes its validated chunked branch (the SIGHET_INFOMAT-unset
            # behavior); the fast multi-shard route needs slot_holder=.
            logger.warning(
                "route_information_matrix: multi-shard holder without "
                "slot_holder -> dropping data_index; the sig-het fast "
                "information matrix falls back to the chunked route "
                "(pass slot_holder=<SubBandBuffer> for the fast path).")
            data_index = None
        # ``data_index`` is only meaningful to the sig-het wrapper, where it
        # is the per-source BUFFER SLOT feeding the in-model reference
        # lookup; raw comps have no such parameter. Forward it ONLY when the
        # caller supplied one, and slice it with the rows -- passing it
        # full-length while ``params_host[pos]`` is a subset would misalign
        # every shard but the first.
        _di = {} if data_index is None else {"data_index": data_index}
        if not cls._is_multi(holder):
            return comp.information_matrix(
                params_phys, holder, inds=inds,
                noise_index=noise_index, **_di, **swap_kwargs)
        xp = holder.xp
        # Speed-diagnosis spans (previously invisible in [GB_TIMING]).
        _rtm = getattr(holder, "_prop_timer", None)
        if _rtm is not None:
            _rtm.count("route_infomat_calls")
        dev_res = _router_device_resident()
        with _tspan(_rtm, "route_host_stage"):
            views = cls._shard_views(holder)
            # Partition by the owning shard of each source's WALKER: the
            # matrix weights by that walker's PSD. (Historically data_index
            # was assumed irrelevant here and aliased to noise_index -- true
            # for the chunked Fisher, false once the sig-het route consumes
            # a slot index.)
            parts = cls._partition(holder, noise_index, noise_index)
            if dev_res:
                params_src = (params_phys if getattr(params_phys, "ndim", 0) == 2
                              else xp.atleast_2d(xp.asarray(params_phys)))
                di_src = (None if data_index is None
                          else np.atleast_1d(np.asarray(asnumpy(data_index))))
            else:
                params_src = np.atleast_2d(asnumpy(params_phys))
                di_src = (None if data_index is None
                          else np.atleast_1d(asnumpy(data_index)))
            num = int(params_src.shape[0])
            take, assemble = cls._make_assembler(num, xp, dev_res)
            items = [
                (si, view, cls._comp_for(comp, holder, view), pos, intra,
                 intra_noise,
                 _slice_rows(xp, params_src, pos),
                 None if di_src is None else di_src[pos])
                for si, (view, (pos, intra, intra_noise))
                in enumerate(zip(views, parts))
                if pos.shape[0]
            ]
        pieces = [None] * len(views)

        def _shard(si, view, comp_s, pos, intra, intra_noise, p_part, di_part):
            with device_context(xp, view.device):
                _di_s = ({} if di_part is None
                         else {"data_index": xp.asarray(di_part)})
                out_s = comp_s.information_matrix(
                    xp.asarray(p_part), view, inds=inds,
                    noise_index=intra if intra_noise is None else intra_noise,
                    **_di_s, **swap_kwargs)
                pieces[si] = (pos, take(out_s))

        with _tspan(_rtm, "route_dispatch"):
            cls._dispatch_shards(holder, items, _shard,
                                 state_ids=[id(it[2]) for it in items])
        with _tspan(_rtm, "route_assemble"):
            return assemble([p for p in pieces if p is not None], 0.0)

    @classmethod
    def _route_infomat_by_slots(cls, comp, slot_holder, params_phys, *,
                                inds, data_index, **swap_kwargs):
        """Slot-shard information-matrix route (sig-het in-model fast leg).

        Partitions sources by the BUFFER holder's slot shard -- the SAME
        ``split_map``/``gpu_splits`` mapping :meth:`setup_in_model` used to
        fan the in-model reference build out -- and hands each shard its
        INTRA-shard rows as both ``data_index`` (the space each per-device
        comp replica's ``_slot_to_ref`` map is keyed in, by construction)
        and ``noise_index``. The latter is unused by the in-model sig-het
        scorer (the walker's invC is baked into the reference stash at
        setup) but stays a VALID per-walker binding on the buffer: slot
        row ``g``'s invC slab was gathered from ``g``'s own walker by the
        buffer fill, so even the comp's chunked fallback branch (knob off /
        reference not armed) scores each source against its own walker's
        PSD. Each shard's call runs against the buffer's persistent
        :class:`_ShardHolderView` inside the owning device context on the
        module-cached device-local comp replica (:meth:`_comp_for` -- the
        same cache ``engine_factory`` replicas resolve through, so the
        stash written at setup is the stash read here).
        """
        xp = slot_holder.xp
        _rtm = getattr(slot_holder, "_prop_timer", None)
        if _rtm is not None:
            _rtm.count("route_infomat_slot_calls")
        dev_res = _router_device_resident()
        with _tspan(_rtm, "route_host_stage"):
            views = cls._shard_views(slot_holder)
            parts = cls._partition(slot_holder, data_index)
            if dev_res:
                params_src = (params_phys
                              if getattr(params_phys, "ndim", 0) == 2
                              else xp.atleast_2d(xp.asarray(params_phys)))
            else:
                params_src = np.atleast_2d(asnumpy(params_phys))
            num = int(params_src.shape[0])
            take, assemble = cls._make_assembler(num, xp, dev_res)
            items = [
                (si, view, cls._comp_for(comp, slot_holder, view), pos,
                 intra, _slice_rows(xp, params_src, pos))
                for si, (view, (pos, intra, _))
                in enumerate(zip(views, parts))
                if pos.shape[0]
            ]
        pieces = [None] * len(views)

        def _shard(si, view, comp_s, pos, intra, p_part):
            with device_context(xp, view.device):
                intra_dev = xp.asarray(intra)
                out_s = comp_s.information_matrix(
                    xp.asarray(p_part), view, inds=inds,
                    noise_index=intra_dev, data_index=intra_dev,
                    **swap_kwargs)
                pieces[si] = (pos, take(out_s))

        with _tspan(_rtm, "route_dispatch"):
            cls._dispatch_shards(slot_holder, items, _shard,
                                 state_ids=[id(it[2]) for it in items])
        with _tspan(_rtm, "route_assemble"):
            return assemble([p for p in pieces if p is not None], 0.0)

    @classmethod
    def route_fstat_ll(cls, comp, method_name, holder, params_phys, *,
                       data_index, noise_index=None, **kwargs):
        """Route a raw F-stat comp entry per shard.

        ``comp`` is the raw single-shard comp (``GBWDMComputations`` /
        ``GBFDComputations``) and ``method_name`` its F-stat entry
        (``"get_fstat_ll_wdm"`` / ``"get_fstat_ll_fd"``): batched over
        binaries, consuming ``holder.linear_data_arr[0]`` and returning
        ``(N (num_bin, 4), M_upper (num_bin, 10))``. Like
        :meth:`route_information_matrix` it runs on the RAW comp (the
        F-stat basis filters are not an engine op), so it gets its own
        classmethod entry rather than the instance router. Single-shard
        holders pass straight through (no overhead). Multi-shard holders
        are partitioned by the owning shard of each binary's walker
        (``data_index``), computed per shard against a persistent
        :class:`_ShardHolderView` inside the owning device context, and
        ``(N, M)`` are reassembled full-length on the caller's device
        (host-routed, no P2P).

        The comp is taken as an OBJECT (not a bound method) so each shard can
        be dispatched to its own device-local replica
        (:meth:`_comp_for`) -- the in-fit F-stat scores every candidate
        against ONE reference walker, so in practice every row lands on that
        walker's shard, which is exactly the case that reads foreign
        device pointers when the comp is shared.
        """
        if not cls._is_multi(holder):
            return getattr(comp, method_name)(
                params_phys, holder, data_index=data_index,
                noise_index=noise_index, **kwargs)
        if data_index is None:
            raise ValueError(
                "route_fstat_ll requires an explicit data_index on "
                "multi-shard holders (the all-zeros default is only "
                "meaningful for a single-shard buffer).")
        if getattr(holder, "slab_min_f", None) is not None:
            # Scope assertion, kept deliberately. ``_ShardHolderView`` now
            # slices ``slab_min_f`` to its own rows, so a slab holder would
            # in fact be handled correctly here -- but the in-fit F-stat runs
            # on the parent residual ACA by design (it scans one walker's
            # FULL residual, not a per-band slab), and a slab holder reaching
            # this entry point means a caller took a path nobody has
            # validated. Fail loudly rather than silently scan narrow slabs.
            raise NotImplementedError(
                "route_fstat_ll does not support narrow per-band slab "
                "holders; F-stat runs on the parent residual ACA, which "
                "carries no slab metadata.")
        xp = holder.xp
        # Speed-diagnosis spans (previously invisible in [GB_TIMING]).
        _rtm = getattr(holder, "_prop_timer", None)
        if _rtm is not None:
            _rtm.count("route_fstat_calls")
        dev_res = _router_device_resident()
        with _tspan(_rtm, "route_host_stage"):
            views = cls._shard_views(holder)
            parts = cls._partition(holder, data_index, noise_index)
            if dev_res:
                params_src = (params_phys if getattr(params_phys, "ndim", 0) == 2
                              else xp.atleast_2d(xp.asarray(params_phys)))
            else:
                params_src = np.atleast_2d(asnumpy(params_phys))
            num = int(params_src.shape[0])
            take, assemble = cls._make_assembler(num, xp, dev_res)
            items = [
                (si, view, cls._comp_for(comp, holder, view), pos, intra,
                 intra_noise,
                 _slice_rows(xp, params_src, pos))
                for si, (view, (pos, intra, intra_noise))
                in enumerate(zip(views, parts))
                if pos.shape[0]
            ]
        N_pieces = [None] * len(views)
        M_pieces = [None] * len(views)

        def _shard(si, view, comp_s, pos, intra, intra_noise, p_part):
            comp_method = getattr(comp_s, method_name)
            with device_context(xp, view.device):
                N_s, M_s = comp_method(
                    xp.asarray(p_part), view,
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    **kwargs)
                N_pieces[si] = (pos, take(N_s))
                M_pieces[si] = (pos, take(M_s))

        with _tspan(_rtm, "route_dispatch"):
            cls._dispatch_shards(holder, items, _shard,
                                 state_ids=[id(it[2]) for it in items])
        with _tspan(_rtm, "route_assemble"):
            return (
                assemble([p for p in N_pieces if p is not None], 0.0),
                assemble([p for p in M_pieces if p is not None], 0.0),
            )

    @classmethod
    def route_sighet_fstat(cls, comp, holder, *, xp, Tobs, f0_lims_hz,
                           data_index, noise_index=None, **build_kwargs):
        """Shard-route the sig-het shared-reference F-stat scorer.

        ``comp`` is the sig-het wrapper (``GBSignalHetComputations``). Its
        F-stat surface is stateful -- ``setup_fstat_references`` folds the
        reference walker's residual into a stash ON the comp that every
        later ``get_fstat_ll_wdm`` scores through -- so unlike
        :meth:`route_fstat_ll` (one routed call per batch) the whole SCORER
        pins to one shard: the F-stat is single-shard by contract (every
        candidate scores against the ONE reference walker ``data_index``),
        making the partition trivial -- that walker's shard owns every
        call. Setup and score both run against the same module-cached
        device-local comp replica (:meth:`_comp_for`), so the stash the
        lazy reference-block builds write is the stash the score calls
        read, on buffers the shard's kernels can legally dereference.

        Returns the ``call_fstat`` closure from
        :func:`lisatools.sampling.fstat_gridfit.build_sighet_call_fstat`
        built against the reference walker's :class:`_ShardHolderView` with
        its INTRA-shard row index; candidates in / ``(N, M)`` out are
        host-routed across the shard's ``device_context`` (no P2P), so the
        adapter and the sweeps never see the sharding. Single-shard holders
        pass straight through (no overhead, no wrapper).

        With TWO OR MORE run devices (``holder.gpus``),
        ``FSTAT_SIGHET_MULTIDEV=1`` fans the scorer out over ALL of them
        (:meth:`_sighet_fstat_multidevice`): the F-stat is single-shard by
        contract, so the pinned form leaves every other GPU idle for the
        whole comb + stage-B fit. OPT-IN (default 0) until the on-GPU
        parity gate passes; ``FSTAT_SIGHET_MULTIDEV=check`` runs the
        fan-out WITH a pinned single-device shadow scorer and hard-compares
        every batch (the on-cluster bisector for the observed GPU
        divergence). CPU and single-GPU holders never take the fan-out
        path at all.
        """
        from ...sampling.fstat_gridfit import build_sighet_call_fstat

        if not cls._is_multi(holder):
            return build_sighet_call_fstat(
                comp, holder, xp=xp, Tobs=Tobs, f0_lims_hz=f0_lims_hz,
                data_index=data_index, noise_index=noise_index,
                **build_kwargs)
        if data_index is None:
            raise ValueError(
                "route_sighet_fstat requires an explicit data_index on "
                "multi-shard holders (the reference walker's row; the "
                "all-zeros default is only meaningful for a single-shard "
                "buffer).")
        if getattr(holder, "slab_min_f", None) is not None:
            # Same scope assertion as route_fstat_ll: the F-stat scans one
            # walker's FULL residual on the parent ACA, never a per-band
            # slab -- a slab holder here means an unvalidated caller path.
            raise NotImplementedError(
                "route_sighet_fstat does not support narrow per-band slab "
                "holders; F-stat runs on the parent residual ACA, which "
                "carries no slab metadata.")
        views = cls._shard_views(holder)
        parts = cls._partition(
            holder, np.atleast_1d(int(data_index)),
            None if noise_index is None else np.atleast_1d(int(noise_index)))
        view, (_pos, intra, intra_noise) = next(
            (v, p) for v, p in zip(views, parts) if p[0].shape[0])
        gpus = getattr(holder, "gpus", None)
        mode = os.environ.get("FSTAT_SIGHET_MULTIDEV", "0")
        if gpus is not None and len(gpus) >= 2 and mode in ("1", "check"):
            # GATE PASSED 2026-08-13 (LAT a86c52af). `=check` on a 2xH100
            # allocation (7.0-7.8 mHz) engaged the fan-out and ran a full
            # comb+stageB fit in 122.3 s / 224 peaks with ZERO diverging
            # batches against the pinned scorer. The fan-out is VALIDATED and
            # the 23-month script has run on it since; `=1` is a supported
            # production setting, not an experiment.
            #
            # History, kept because it explains why this is opt-in rather
            # than the default: the first on-GPU run (2026-08-12) produced
            # grids that DIFFERED from the single-device path (F_max rel up
            # to 0.81, best-sky sign flips) while the CPU fake/real-comp
            # bit-identity tests passed. The 2026-08-11 code audit excluded
            # the merge bookkeeping (disjoint host row ranges), the transfer
            # ordering (every lane D2H is a blocking .get() behind the wraps'
            # own cudaDeviceSynchronize) and the holder row layouts; the
            # kernel wraps hold the GIL, so lanes cannot race in C++ either.
            # What CPU could not reach was the on-GPU scoring of the
            # non-primary lane's comp replica -- and that divergence was
            # closed by the drift-campaign replica fixes (template twin
            # 5d01095, orbits replica 7d6fd4c, WDMSettings t0 319782c),
            # which is what the =check gate then confirmed.
            #
            # `=check` remains available: it shadows every batch with the
            # pinned scorer and fails loudly on the first diverging row,
            # localizing it to a lane. Re-run it after any change to the
            # sharding, the replica construction, or the merge.
            return cls._sighet_fstat_multidevice(
                comp, holder, view, int(intra[0]),
                int(intra[0] if intra_noise is None else intra_noise[0]),
                xp=xp, Tobs=Tobs, f0_lims_hz=f0_lims_hz,
                check=(mode == "check"), **build_kwargs)
        comp_s = cls._comp_for(comp, holder, view)
        inner = build_sighet_call_fstat(
            comp_s, view, xp=xp, Tobs=Tobs, f0_lims_hz=f0_lims_hz,
            data_index=int(intra[0]),
            noise_index=(None if intra_noise is None
                         else int(intra_noise[0])),
            **build_kwargs)

        def call_fstat(params):
            # Candidates in / (N, M) out are host-routed; the scorer --
            # including the lazy reference-block builds it triggers -- runs
            # inside the owning shard's device context.
            params_host = np.atleast_2d(asnumpy(params))
            with device_context(holder.xp, view.device):
                N_s, M_s = inner(xp.asarray(params_host))
                N_host, M_host = asnumpy(N_s), asnumpy(M_s)
            return xp.asarray(N_host), xp.asarray(M_host)

        return call_fstat

    @classmethod
    def _sighet_fstat_multidevice(cls, comp, holder, view, intra_data,
                                  intra_noise, *, xp, Tobs, f0_lims_hz,
                                  check=False, **build_kwargs):
        """All-device fan-out for the sig-het F-stat scorer.

        REPLICATE, don't partition, the resident state: every run device
        gets its own comp replica (module-cached ``_comp_for``) and its own
        :class:`_FStatRefRowHolder` -- the reference walker's residual +
        inverse-PSD rows copied onto that device ONCE here (host-routed, no
        P2P) -- and builds its OWN copy of whichever ~GB reference block its
        rows need (``build_sighet_call_fstat``'s lazy ``_ensure_block``,
        whose per-lane state dict is independent). Reference builds are
        deterministic, so concurrent per-device builds are free
        parallelization, and copying the rows once at adapter build means
        every lane's blocks fold the SAME residual snapshot.

        Each ``call_fstat`` batch is row-split into near-equal CONTIGUOUS
        chunks (the sweeps order rows f0-locally -- node-major comb,
        f0-sorted stage-B boxes -- so contiguous chunks stay
        reference-block coherent per lane). Lanes run concurrently on host
        threads (one per device with rows; cupy's current device is
        thread-local; the fstat scorer + reference-build wraps release the
        GIL via ``nb::call_guard`` in binding_gbgpu -- a GBGPU wheel built
        before that guard still works but serializes the lanes' kernel
        calls), each entering its own device context and returning host
        ``(N, M)``.

        DETERMINISM: results are IDENTICAL to the single-device path --
        every row is scored by exactly ONE lane whose replica performs the
        same arithmetic on the same folded inputs as the pinned scorer
        (replica bit-identity is pinned by
        ``SigHetFStatRouteRealCompTest``), the per-row |df0| hard-assert
        runs unchanged inside every lane, and the merge below is a pure
        permutation into disjoint host row ranges -- no reductions ever
        cross devices.

        ``check=True`` (``FSTAT_SIGHET_MULTIDEV=check``): every batch is
        ALSO scored through the pinned single-device scorer (the exact
        else-branch construction) and hard-compared bit-for-bit. On the
        first divergence it logs per-lane forensics -- which lane's row
        range diverges, on which device, through the prototype comp or a
        replica -- and raises. This is the on-cluster bisector for the
        observed GPU divergence: a mismatch confined to the non-primary
        lane's range convicts that lane's comp replica; a mismatch pattern
        crossing lane boundaries convicts the merge/transfer machinery.
        The pinned shadow may SHARE a comp with the walker-shard lane;
        ``build_sighet_call_fstat``'s stash-identity guard makes the two
        closures rebuild instead of scoring a foreign block.
        """
        from ...sampling.fstat_gridfit import build_sighet_call_fstat

        n_slabs = int(view.acs_total_entries)
        # Slice the walker's rows ON the owning device, host only the rows
        # (asnumpy of the whole shard buffer would round-trip every walker).
        with device_context(holder.xp, view.device):
            data_row_host = np.ascontiguousarray(asnumpy(
                xp.asarray(view.linear_data_arr[0]).reshape(
                    n_slabs, -1)[int(intra_data)]))
            psd_row_host = np.ascontiguousarray(asnumpy(
                xp.asarray(view.linear_psd_arr[0]).reshape(
                    n_slabs, -1)[int(intra_noise)]))

        lanes = []
        lane_comps = []
        for dev in [int(g) for g in holder.gpus]:
            with device_context(holder.xp, dev):
                ref_holder = _FStatRefRowHolder(
                    holder, dev,
                    xp.asarray(data_row_host), xp.asarray(psd_row_host))
            comp_d = cls._comp_for(comp, holder, ref_holder)
            lane_comps.append(comp_d)
            lanes.append((dev, build_sighet_call_fstat(
                comp_d, ref_holder, xp=xp, Tobs=Tobs,
                f0_lims_hz=f0_lims_hz, data_index=0, noise_index=0,
                **build_kwargs)))
        logger.info(
            "[sighet-fstat] multi-device scorer: %d lanes on devices %s%s "
            "(FSTAT_SIGHET_MULTIDEV=0 restores the single-device pin)",
            len(lanes), [dev for dev, _ in lanes],
            " + pinned shadow CHECK" if check else "")

        if check:
            # The exact pinned construction (the route's else branch): the
            # walker-shard replica scoring against the live shard view.
            comp_pin = cls._comp_for(comp, holder, view)
            inner_pin = build_sighet_call_fstat(
                comp_pin, view, xp=xp, Tobs=Tobs, f0_lims_hz=f0_lims_hz,
                data_index=int(intra_data), noise_index=int(intra_noise),
                **build_kwargs)
        else:
            comp_pin = inner_pin = None

        def _check_batch(params_host, N_host, M_host):
            """Shadow-score the batch on the pinned scorer; fail loudly on
            the first diverging row with lane-resolved forensics."""
            with device_context(holder.xp, view.device):
                N_ref, M_ref = inner_pin(xp.asarray(params_host))
                N_ref, M_ref = asnumpy(N_ref), asnumpy(M_ref)
            if np.array_equal(N_host, N_ref) and np.array_equal(M_host, M_ref):
                return
            n = int(params_host.shape[0])
            bounds = (n * np.arange(len(lanes) + 1)) // len(lanes)
            bad = (np.any(N_host != N_ref, axis=1)
                   | np.any(M_host != M_ref, axis=1))
            lines = []
            for i, (dev, _inner) in enumerate(lanes):
                s, e = int(bounds[i]), int(bounds[i + 1])
                nb = int(bad[s:e].sum())
                dN = np.abs(N_host[s:e] - N_ref[s:e])
                dM = np.abs(M_host[s:e] - M_ref[s:e])
                if lane_comps[i] is comp_pin:
                    kind = "PINNED-SHARED"
                elif lane_comps[i] is comp:
                    kind = "prototype"
                else:
                    kind = "replica"
                lines.append(
                    f"lane {i} dev {dev} rows [{s}:{e}) comp={kind} "
                    f"bad {nb}/{e - s} maxdN {dN.max() if dN.size else 0:.3e} "
                    f"maxdM {dM.max() if dM.size else 0:.3e}")
            first = int(np.argmax(bad))
            msg = ("[sighet-fstat] MULTIDEV CHECK FAILED: fan-out != pinned "
                   f"scorer on {int(bad.sum())}/{n} rows (first bad row "
                   f"{first}, f0={params_host[first, 1]:.9e} Hz).\n  "
                   + "\n  ".join(lines))
            logger.error(msg)
            raise RuntimeError(msg)

        def call_fstat(params):
            params_host = np.atleast_2d(asnumpy(params))
            n = int(params_host.shape[0])
            bounds = (n * np.arange(len(lanes) + 1)) // len(lanes)
            N_host = np.zeros((n, 4), dtype=np.float64)
            M_host = np.zeros((n, 10), dtype=np.float64)

            def _score(i):
                s, e = int(bounds[i]), int(bounds[i + 1])
                dev, inner = lanes[i]
                with device_context(holder.xp, dev):
                    N_s, M_s = inner(xp.asarray(params_host[s:e]))
                    # permutation merge into disjoint row ranges (see the
                    # determinism note in the method docstring)
                    N_host[s:e] = asnumpy(N_s)
                    M_host[s:e] = asnumpy(M_s)

            active = [i for i in range(len(lanes))
                      if bounds[i + 1] > bounds[i]]
            # thread_pool only touched with >1 populated lane (the real
            # ACA property allocates the pool on first access)
            pool = (getattr(holder, "thread_pool", None)
                    if len(active) > 1 else None)
            if pool is not None:
                futures = [pool.submit(_score, i) for i in active]
                for f in futures:
                    f.result()  # re-raise lane exceptions in caller
            else:
                for i in active:
                    _score(i)
            if inner_pin is not None:
                _check_batch(params_host, N_host, M_host)
            return xp.asarray(N_host), xp.asarray(M_host)

        return call_fstat

    @classmethod
    def _lane_weight_spec(cls):
        """``GB_FSTAT_NM_LANE_WEIGHTS`` as given (None when unset/blank)."""
        v = os.environ.get("GB_FSTAT_NM_LANE_WEIGHTS", "").strip()
        return v or None

    @classmethod
    def make_fstat_nm_lanes(cls, comp, method_name, holder, walker_ref,
                            *, check=False, timer=None, **kwargs):
        """All-device fan-out for the per-row F-stat ``(N, M)`` scorer.

        Production autopsy (v7 snapshot 2, 2026-08-27):
        :meth:`route_fstat_ll` partitions rows by walker, and the per-row
        center chain stamps EVERY row with the ONE reference walker -- so
        the whole leg (documented at ~735 s/propose, "half the rj black
        box") ran serially on that walker's device while the other GPU
        idled (43% of gb_search telemetry samples single-GPU).

        Same design as :meth:`_sighet_fstat_multidevice` (REPLICATE, don't
        partition): the reference walker's residual + inverse-PSD rows are
        copied onto every run device ONCE here (host-routed, no P2P) as a
        :class:`_FStatRefRowHolder`, each device gets its own comp replica
        (:meth:`_comp_for`), and each batch splits into near-equal
        CONTIGUOUS row lanes scored concurrently (host threads; cupy's
        current device is thread-local). The merge is a pure permutation
        into disjoint host row ranges, and every lane's replica performs
        the same arithmetic on the same snapshot rows as the pinned
        scorer, so results are identical to the single-device path.
        This leg is READ-ONLY (it never writes a residual), and the
        parent residual only changes at unit open/close -- a snapshot
        taken at unit start equals what the pinned per-round call reads.

        ``check=True`` (``GB_FSTAT_NM_MULTIDEV=check``): every batch is
        ALSO scored through the exact pinned single-device path (the
        owning shard's view + intra index) and compared bit-for-bit,
        with per-lane forensics on the first divergence -- the
        on-cluster parity gate, mirroring ``FSTAT_SIGHET_MULTIDEV=check``.

        ``timer``: optional ``gbspecialstretch._ProposeTimer``. The returned
        ``call_NM`` then reports two sub-spans inside the move's
        ``fstat_nm_lanes`` stage --

        * ``fstat_nm_h2d`` -- ``asnumpy(params_phys)``. A FORCED DEVICE
          SYNC, so under ``GB_PROP_TIMING_SYNC=0`` it can absorb kernels
          queued before it (the transform, the caller's gathers) and read
          high; under ``=1``/``=all`` it is the copy. Worth isolating
          regardless: at 4.55 M candidate rows per propose this leg
          host-stages every one of them.
        * ``fstat_nm_lane_score`` -- the threaded per-device scoring and its
          own per-lane D2H merge. This one is a genuine blocking join in
          BOTH modes (the futures are awaited), so it is the closest thing
          to an honest sync-off measurement in the whole centre chain.

        Only the main thread is timed: the per-lane bodies run in the holder
        thread pool and accumulating into a shared dict from several threads
        would race.

        Returns ``call_NM(params_phys) -> (N, M)`` on the caller's array
        module, or ``None`` for single-shard holders (caller keeps the
        pinned path).
        """
        if not cls._is_multi(holder):
            return None
        xp = holder.xp
        split_map, intra_map = shard_lookup_maps(holder)
        si = int(split_map[int(walker_ref)])
        intra = int(intra_map[int(walker_ref)])
        view = cls._shard_views(holder)[si]
        n_slabs = int(view.acs_total_entries)
        with device_context(holder.xp, view.device):
            data_row_host = np.ascontiguousarray(asnumpy(
                xp.asarray(view.linear_data_arr[0]).reshape(
                    n_slabs, -1)[intra]))
            psd_row_host = np.ascontiguousarray(asnumpy(
                xp.asarray(view.linear_psd_arr[0]).reshape(
                    n_slabs, -1)[intra]))

        lanes = []
        for dev in [int(g) for g in holder.gpus]:
            with device_context(holder.xp, dev):
                ref_holder = _FStatRefRowHolder(
                    holder, dev,
                    xp.asarray(data_row_host), xp.asarray(psd_row_host))
            lanes.append((dev, cls._comp_for(comp, holder, ref_holder),
                          ref_holder))
        # Row-lane weighting (GB_FSTAT_NM_LANE_WEIGHTS, default equal =
        # bit-identical to the historical 50/50 split). Resolved ONCE here,
        # not per batch: call_NM runs ~1,409 times per propose and an
        # os.environ read per call is pure overhead on the hot path.
        _lane_w = cls._lane_weight_spec()
        _probe = fstat_nm_lane_bounds(len(lanes) * 1000, len(lanes), _lane_w)
        logger.info(
            "[fstat-NM] multi-device per-row scorer: %d lanes on devices "
            "%s for walker_ref=%d%s, row split %s (GB_FSTAT_NM_MULTIDEV=0 "
            "restores the single-device pin)", len(lanes),
            [d for d, _, _ in lanes], int(walker_ref),
            " + pinned shadow CHECK" if check else "",
            ("equal" if _lane_w is None else
             f"GB_FSTAT_NM_LANE_WEIGHTS={_lane_w!r} -> "
             f"{list(np.diff(_probe) / (len(lanes) * 10.0))}%"))

        comp_pin = cls._comp_for(comp, holder, view) if check else None

        def _pinned(params_host):
            n = int(params_host.shape[0])
            with device_context(holder.xp, view.device):
                idx = xp.full(n, intra, dtype=xp.int32)
                N_ref, M_ref = getattr(comp_pin, method_name)(
                    xp.asarray(params_host), view,
                    data_index=idx, noise_index=idx, **kwargs)
                return asnumpy(N_ref), asnumpy(M_ref)

        def call_NM(params_phys):
            with _tspan(timer, "fstat_nm_h2d"):
                params_host = np.atleast_2d(asnumpy(params_phys))
            n = int(params_host.shape[0])
            bounds = fstat_nm_lane_bounds(n, len(lanes), _lane_w)
            N_host = np.zeros((n, 4), dtype=np.float64)
            M_host = np.zeros((n, 10), dtype=np.float64)
            # Per-lane wall times for THIS batch. Each lane thread writes
            # only its own slot (no shared-dict race -- the reason the
            # docstring gives for not timing lane bodies stops at dicts);
            # the MAIN thread folds them into the timer after the join.
            # Honest per lane: _score ends with its own D2H (asnumpy), so
            # the lane's queued kernels are drained inside its own span.
            # This is the instrument the GPU0-35%/GPU1-71% imbalance needs:
            # equal-count lanes with unequal times pin the asymmetry to
            # row cost / device contention, and the time ratio is directly
            # the GB_FSTAT_NM_LANE_WEIGHTS vector to arm.
            lane_secs = [0.0] * len(lanes)

            def _score(i):
                t0 = time.perf_counter()
                s, e = int(bounds[i]), int(bounds[i + 1])
                dev, comp_d, rh = lanes[i]
                with device_context(holder.xp, dev):
                    zeros = xp.zeros(e - s, dtype=xp.int32)
                    N_s, M_s = getattr(comp_d, method_name)(
                        xp.asarray(params_host[s:e]), rh,
                        data_index=zeros, noise_index=zeros, **kwargs)
                    # permutation merge into disjoint host row ranges
                    N_host[s:e] = asnumpy(N_s)
                    M_host[s:e] = asnumpy(M_s)
                lane_secs[i] = time.perf_counter() - t0

            active = [i for i in range(len(lanes))
                      if bounds[i + 1] > bounds[i]]
            pool = (getattr(holder, "thread_pool", None)
                    if len(active) > 1 else None)
            with _tspan(timer, "fstat_nm_lane_score"):
                if pool is not None:
                    futures = [pool.submit(_score, i) for i in active]
                    for f in futures:
                        f.result()  # re-raise lane exceptions in caller
                else:
                    for i in active:
                        _score(i)
            if timer is not None and len(active) > 1:
                stages = getattr(timer, "stages", None)
                count = getattr(timer, "count", None)
                for i in active:
                    key = f"fstat_nm_lane{i}_dev{lanes[i][0]}"
                    if stages is not None:
                        stages[key] = stages.get(key, 0.0) + lane_secs[i]
                    if count is not None:
                        count(f"fstat_nm_lane{i}_rows",
                              int(bounds[i + 1]) - int(bounds[i]))
            if comp_pin is not None:
                N_ref, M_ref = _pinned(params_host)
                if not (np.array_equal(N_host, N_ref)
                        and np.array_equal(M_host, M_ref)):
                    bad = (np.any(N_host != N_ref, axis=1)
                           | np.any(M_host != M_ref, axis=1))
                    lines = []
                    for i in range(len(lanes)):
                        s, e = int(bounds[i]), int(bounds[i + 1])
                        nb = int(bad[s:e].sum())
                        lines.append(
                            f"lane {i} dev {lanes[i][0]} rows [{s}:{e}) "
                            f"bad {nb}/{e - s}")
                    msg = ("[fstat-NM] MULTIDEV CHECK FAILED: fan-out != "
                           f"pinned scorer on {int(bad.sum())}/{n} rows.\n  "
                           + "\n  ".join(lines))
                    logger.error(msg)
                    raise RuntimeError(msg)
            return xp.asarray(N_host), xp.asarray(M_host)

        return call_NM


def make_routed_band_engine(basis_settings, *, xp, gb_wdm_comp=None,
                            gb_fd_comp=None, **engine_kwargs):
    """Build the shard-routed band likelihood engine for a holder.

    The prototype engine is exactly the
    :func:`gbgpu.gb_likelihood.make_band_likelihood_engine` product the two
    construction sites (:class:`SubBandBuffer` and the move-level parent-ACA
    engine in ``gbspecialstretch``) built before, so single-GPU behaviour is
    unchanged. The added ``engine_factory`` closure rebuilds the SAME engine
    around per-device comp replicas
    (``source_runtime._device_local_gb_comp``) the first time a shard lands
    on a device other than the comps' own -- giving that shard device-local
    chunk geometry / window / orbit + TDI wraps, and, under
    ``GB_SIGHET_INMODEL``, its own heterodyne reference stash.

    Replicas are module-cached and allocate-once; they deliberately do NOT
    follow the holder's proposal lifetime (rebuilding a ``GBTDIonTheFly`` per
    proposal would be ruinous) and never reach the settings tree.
    """
    from gbgpu.gb_likelihood import make_band_likelihood_engine

    def _engine_factory(device, primary):
        from ..stock.erebor.source_runtime import _device_local_gb_comp

        with device_context(xp, device):
            return make_band_likelihood_engine(
                basis_settings,
                gb_wdm_comp=_device_local_gb_comp(
                    gb_wdm_comp, xp, device, primary),
                gb_fd_comp=_device_local_gb_comp(
                    gb_fd_comp, xp, device, primary),
                **engine_kwargs,
            )

    return _RoutedBandEngine(
        make_band_likelihood_engine(
            basis_settings, gb_wdm_comp=gb_wdm_comp, gb_fd_comp=gb_fd_comp,
            **engine_kwargs),
        engine_factory=_engine_factory,
    )


class SubBandBuffer(AnalysisContainerArray, LISAToolsParallelModule):
    """Per-(temp, walker, band) scratch buffers for the GB special moves.

    One :class:`AnalysisContainer` per active cell, all owned directly by
    this object (it *is* the :class:`AnalysisContainerArray`): the linear
    data buffer holds each cell's residual window, the linear PSD buffer the
    matching inverse-PSD slice. GB sources are written into / removed from
    the residual through the domain-aware likelihood engine
    (:meth:`add_sources_to_band_buffer` / :meth:`remove_sources_from_band_buffer`),
    so the inner MCMC loop avoids reallocating large arrays each iteration.

    The most-used members are:

    - ``special_indices_unique`` / ``special_indices_unique_sort``: lookup
      tables that map a per-source ``special_index`` back into the buffer
      ordering.
    - ``params_interest``: parameters of GBs that participate in the move.

    Sign convention: each cell buffer holds the **residual**
    ``data - sum(templates)``. Removing a source from the model therefore
    means *adding* its template back into the buffer (``factor=+1``);
    adding a source to the model subtracts it (``factor=-1``).
    """

    @property
    def xp(self) -> Union[ModuleType, numpy, cupy]:
        """Active array module (NumPy or CuPy) for this buffer.

        Overrides the ``gpus``-based :class:`AnalysisContainerArray`
        property: the buffer's backend decides the array module, so a
        CUDA-backend run on the current device (``gpus=None``) still
        allocates on the GPU.
        """
        return self.backend.xp

    @property
    def df(self):
        """Frequency spacing used for band-index math.

        FD: the FD bin width. WDM: ``layer_df`` so ``band_edges / df``
        yields WDM layer indices. Overrides the read-only
        :class:`AnalysisContainerArray` property (which derives ``df``
        from ``f_arr``).
        """
        return self._df

    @classmethod
    def supported_backends(cls):
        """List the GPU backend names this buffer supports."""
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_index(self, special_inds_test):
        """Map a special-index test value to its position inside the buffer."""
        # Fixed-capacity resize contract: between ``resize_to(k)`` and the
        # mandatory FULL ``update_special_indices`` rebind the specials map
        # holds placeholders — a searchsorted over it would silently return
        # garbage slots (the stale-specials hazard). Hard error instead.
        assert not getattr(self, "_specials_placeholder", False), (
            "SubBandBuffer.get_index called after resize_to() but before the "
            "mandatory full update_special_indices rebind — the special-"
            "indices map is invalid (placeholder) in this window."
        )
        # Array module from the OPERANDS, never the module-level ``cp``: on
        # a CPU-backend run on a machine where cupy imports (cluster), the
        # module-level ``cp`` is cupy while these are numpy --
        # cupy.searchsorted then raises "Only int or ndarray are supported
        # for a". Same trap as get_buffer's sorter xp (see :2860).
        xp = get_array_module(self.special_indices_unique)
        now_index = (
            self.special_indices_unique_sort[
                xp.searchsorted(
                    self.special_indices_unique[self.special_indices_unique_sort],
                    xp.asarray(special_inds_test),
                    side="right",
                )
                - 1
            ]
        ).astype(xp.int32)
        return now_index

    def __init__(
        self,
        is_rj,
        nwalkers,
        gb,
        band_edges,
        band_N_vals,
        unique_band_combos,
        params_interest,
        num_bands_now,
        nchannels,
        data_length,
        special_indices_unique,
        transform_fn,
        waveform_kwargs,
        df,
        sources_now_map,
        sources_inject_now_map,
        special_band_inds,
        opt_snr_rej_samp_limit=None,
        snr_rej_detected=False,
        force_backend="gpu",
        use_template_arr=False,
        basis_settings: Optional[DomainSettingsBase] = None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        keep_sens_mat: bool = False,
        wdm_band_slab_layers: Optional[int] = None,
        wdm_slab_guard_layers: int = 1,
        alloc_capacity: Optional[int] = None,
        *args,
        **kwargs,
    ):
        # (The gb_likelihood import is deferred inside
        # ``make_routed_band_engine`` -- gb_likelihood imports nothing from
        # here, but the module import graph stays acyclic if that changes.)
        self.force_backend = force_backend
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        assert self.backend.name.split("_")[-1] == gb.backend.name.split("_")[-1]
        self.gb = gb
        # Domain computation objects (gbgpu.gbcomps.GBWDMComputations /
        # GBFDComputations prototype). The one matching ``basis_settings``
        # is required; the other stays None. The legacy ``gb`` handle is
        # kept only for the info-matrix proposal shaping.
        self.gb_wdm_comp = gb_wdm_comp
        self.gb_fd_comp = gb_fd_comp
        self._df = df
        self.nwalkers = nwalkers
        self.sources_now_map, self.sources_inject_now_map = (
            sources_now_map,
            sources_inject_now_map,
        )
        self.band_edges, self.unique_band_combos = band_edges, unique_band_combos
        self.num_bands = len(self.band_edges) - 1
        self.params_interest = params_interest
        self.num_bands_now, self.nchannels, self.data_length = (
            num_bands_now,
            nchannels,
            data_length,
        )
        # Fixed-capacity slot allocation (user ruling 2026-08-14): when
        # ``alloc_capacity`` covers the current binding, EVERY per-slot
        # allocation (the ACA linear data/PSD buffers — and the template
        # twin's — plus the FD per-slot window-start store) is sized at
        # ``alloc_capacity`` slots and only the first ``num_bands_now`` are
        # BOUND. :meth:`resize_to` then rebinds a later unit's k <= capacity
        # cells into the FRONT of the same allocation instead of dropping
        # and rebuilding a multi-GB buffer per unit. Leading-axis slices of
        # the C-contiguous slabs are contiguous views sharing memory, so
        # the nanobind kernels (which receive base pointers per call) and
        # the cached engine bindings stay valid across resizes.
        # ``None`` (default) is BIT-IDENTICAL to the pre-capacity behavior.
        # NOTE ``>= num_bands_now`` (not ``>``): a unit bound at exactly
        # capacity must still be capacity-active so a later smaller unit
        # can resize it down instead of rebuilding.
        if alloc_capacity is not None:
            alloc_capacity = int(alloc_capacity)
            if alloc_capacity < int(num_bands_now):
                raise ValueError(
                    f"alloc_capacity={alloc_capacity} is smaller than the "
                    f"initial binding num_bands_now={int(num_bands_now)}."
                )
        self.alloc_capacity = alloc_capacity
        # resize_to leaves placeholder specials until the caller's full
        # update_special_indices rebind lands (see resize_to's contract).
        self._specials_placeholder = False
        # FD store length of one cell window; kept distinct from the ACA's
        # ``data_length`` layout attribute (see :attr:`_fd_store_length`).
        self._fd_store_length_value = data_length
        self.band_N_vals = self.xp.asarray(band_N_vals)
        # TODO: adjust this
        self.edge_buffer = 2000
        self.is_rj = is_rj

        # Frequency-clipped parent domains (FDSettings with min/max_freq)
        # store only bins [ind_min, ind_max]. Every per-cell FD window (and
        # its start index) must live inside that range: clamp the window
        # length here -- BEFORE the ``special_indices_unique`` setter below
        # computes start indices and before the buffers are allocated -- and
        # record the parent bounds for the start-index clamp in the setter.
        # WDM and legacy full-grid FD paths are untouched.
        self._parent_ind_min = None
        self._parent_stored_len = None
        if (
            isinstance(basis_settings, FDSettings)
            and getattr(basis_settings, "ind_min", None) is not None
            and getattr(basis_settings, "ind_max", None) is not None
        ):
            self._parent_ind_min = int(basis_settings.ind_min)
            self._parent_stored_len = (
                int(basis_settings.ind_max) - int(basis_settings.ind_min) + 1
            )
            if self._fd_store_length_value > self._parent_stored_len:
                logger.warning(
                    "FD cell window (%d bins) exceeds the clipped parent domain "
                    "(%d stored bins); clamping each cell window to the full "
                    "stored band.",
                    self._fd_store_length_value,
                    self._parent_stored_len,
                )
                self._fd_store_length_value = self._parent_stored_len

        self.special_indices_unique = special_indices_unique
        self.transform_fn = transform_fn
        # Container-derived roles: phi0's sampled column (phase-max rotation)
        # and whether the container holds per-leaf fills (Eryn per-leaf
        # fill_dict) -- in that case every sampling->physical conversion in
        # this buffer needs the per-row ``leaf_inds``.
        _ib = list(getattr(transform_fn, "input_basis", []) or [])
        self._phi0_col = _ib.index("phi0") if "phi0" in _ib else 3
        self._per_leaf_fill = getattr(transform_fn, "n_leaf_fills", None) is not None
        self.waveform_kwargs = waveform_kwargs
        # None (the GB default path — VGB passes an explicit float and is
        # unaffected) resolves via GB_OPT_SNR_LIMIT, default 5.0. The
        # 2026-08-26 highf forensics measured the SNR-5 floor feeding a
        # hot-ladder noise balloon (83->186 leaves, 19/32 bands occupied
        # at T1-T10) whose at-cap cells then blockaded cold births; the
        # standing F-stat peak-floor ruling ("SNR 5 = noise, keep 8")
        # has this birth-floor counterpart. Probe scripts pin 8.
        if opt_snr_rej_samp_limit is None:
            opt_snr_rej_samp_limit = float(
                os.environ.get("GB_OPT_SNR_LIMIT", "5.0"))
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit
        self.snr_rej_detected = bool(snr_rej_detected)
        self.use_template_arr = use_template_arr
        # Per-band sensitivity storage. Only the inverse covariance ``invC``
        # feeds the WDM / FD likelihood kernels (the ACA repacks
        # ``sens_mat.invC`` into ``linear_psd_arr``; nothing on the band-move
        # path reads the forward ``sens_mat`` array). Default ``False``
        # therefore stores ONLY ``invC`` per band and backs the forward
        # ``sens_mat`` with a zero-storage, shape-correct broadcast view --
        # halving the per-band cross-channel (XYZ 3x3) memory. Set
        # ``keep_sens_mat=True`` to allocate the forward matrix too (e.g. for
        # diagnostics that inspect the per-band covariance directly).
        self.keep_sens_mat = keep_sens_mat
        # Task-b: narrow per-band WDM slabs. Each per-band buffer slab spans a
        # limited number of WDM frequency layers centered on the band --
        # instead of the FULL active band ``Nf_active`` -- cutting the dominant
        # sub-band-buffer memory term by ~Nf_active/slab_Nf. Requires the C++
        # chunked-het kernels built with the task-b per-slab layer origin
        # (default backend); the per-slot origins flow via ``slab_min_f``.
        #   ``wdm_band_slab_layers`` semantics:
        #     None -> OFF (full active band; bit-identical to pre-task-b).
        #     0    -> AUTO: band layer span + 2*(leakage + guard), where
        #             leakage = _WDM_SLAB_LEAKAGE_LAYERS (2) and guard =
        #             ``wdm_slab_guard_layers`` (default 1) -> band_span + 6.
        #     N>0  -> EXPLICIT: exactly N layers (power users).
        # FD path ignores this (already per-band narrow via max_data_store_size).
        self._wdm_band_slab_layers = (
            int(wdm_band_slab_layers) if wdm_band_slab_layers is not None else None
        )
        self._wdm_slab_guard_layers = int(wdm_slab_guard_layers)

        self.tdi_channel_setup = self.waveform_kwargs.get("tdi_channel_setup")
        if self.tdi_channel_setup == "XYZ":
            assert self.nchannels == 3
        else:
            assert "A" in self.tdi_channel_setup and "E" in self.tdi_channel_setup
            logger.warning("using AE(T) channels where we assume ortogonality. This may not be sufficient for realistic orbtis.")

        # Resolve the parent basis-domain settings. Defaults to an FD grid
        # consistent with the legacy Buffer behavior (data_length bins on the
        # parent's df). When invoked via BandSorter.get_buffer, the parent
        # AnalysisContainerArray's settings are forwarded so this buffer can
        # branch on the actual domain (FD vs WDM).
        if basis_settings is None:
            basis_settings = FDSettings(
                N=self.data_length,
                df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
            )
        self._basis_settings = basis_settings

        # Build the per-cell AnalysisContainers and initialise *ourselves* as
        # the AnalysisContainerArray that owns them. On the WDM path the ACA
        # layout metadata (``data_length = Nf_active * Nt_active``,
        # ``end_shape``) replaces the FD-style ctor value -- the inherited
        # linear-buffer indexing needs the ACA meaning; nothing on the WDM
        # move path consumes the FD-style value.
        ac_list, aca_kwargs = self._build_band_ac_list()
        AnalysisContainerArray.__init__(self, ac_list, **aca_kwargs)
        if self.use_template_arr:
            # Templates mirror the band-buffer layout in a twin ACA so they
            # share the same managed memory region. The per-band sensitivity
            # slot on the template ACA is unused but keeps construction
            # symmetric across the two buffers.
            template_ac_list, template_aca_kwargs = self._build_band_ac_list()
            self._acs_template_buffer = AnalysisContainerArray(
                template_ac_list, **template_aca_kwargs
            )

        # psd_shape is exposed for back-compat with downstream consumers that
        # inspect it; it tracks the shape of the per-band PSD view.
        self.psd_shape = (self.num_bands_now,) + self._per_band_sens_shape

        # Geometry line: the buffer is the dominant per-proposal allocation
        # (cells x per-cell slab); one INFO read diagnoses an OOM-scale
        # configuration (e.g. many bands with slab slicing off).
        _cell_mb = (
            np.prod(self._per_band_data_shape)
            * np.dtype(self._per_band_data_dtype).itemsize / 1e6
        )
        _n_copies = 2 if self.use_template_arr else 1
        _pool = ""
        if self.backend.uses_cupy:
            _mp = cp.get_default_memory_pool()
            _pool = (f"  [GPU pool used {_mp.used_bytes() / 1e9:.2f} / "
                     f"total {_mp.total_bytes() / 1e9:.2f} GB]")
        # Host watermark alongside: SIGKILL-class failures are host-side --
        # this line is the last breadcrumb before a cgroup OOM kill.
        import resource as _resource
        import sys as _sys

        _rss_kb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        _rss_gb = _rss_kb / (1e9 if _sys.platform == "darwin" else 1e6)
        _pool += f"  [host maxRSS {_rss_gb:.1f} GB]"
        # HONEST total: the data slab alone under-reports by ~4x for XYZ
        # (invC is (nc, nc, slab) per slot -- the term that actually sized
        # the job-183 OOM).
        _invc_mb = (self.nchannels
                    * float(np.prod(self._per_band_data_shape))
                    * np.dtype(self._per_band_data_dtype).itemsize / 1e6)
        # Memory geometry follows the ALLOCATED slot count (capacity when
        # fixed-capacity is active), not the bound count.
        _n_geom = self._n_slots_alloc
        logger.info(
            "SubBandBuffer: %d cells%s x %s per-cell (%s) ~ %.0f MB data "
            "+ ~%.0f MB invC = ~%.1f GB total%s [band_slab_Nf=%s]%s",
            _n_geom,
            ("" if self.alloc_capacity is None
             else f" ({self.alloc_capacity}-slot alloc, "
                  f"{self.num_bands_now} bound)"),
            tuple(self._per_band_data_shape),
            np.dtype(self._per_band_data_dtype).name,
            _n_copies * _n_geom * _cell_mb,
            _n_geom * _invc_mb,
            (_n_copies * _n_geom * _cell_mb + _n_geom * _invc_mb) / 1e3,
            " (incl. template twin)" if self.use_template_arr else "",
            self.band_slab_Nf, _pool,
        )

        # Build the domain-aware likelihood engine. Dispatch is on
        # ``isinstance(basis_settings, ...)`` -- no string-level mode flag.
        # The engine takes an AnalysisContainerArray at call time, so the
        # buffer's get_swap_ll / get_ll / adjust_sources_in_band_buffer
        # methods don't reach into self.gb (or self.gb_wdm_comp) directly.
        # Persistent per-slot window-start store. Bound BY POINTER into the
        # FD computations clone (FDDomain.start_inds), so it must be updated
        # in place on cell swaps -- never rebound (see the
        # special_indices_unique setter).
        if isinstance(self._basis_settings, FDSettings):
            _starts = self.xp.ascontiguousarray(
                self.xp.asarray(self.start_freq_inds), dtype=self.xp.int32
            ).copy()
            if self.alloc_capacity is not None:
                # Fixed capacity: the pointer-stable per-slot window-start
                # store is allocated at FULL capacity (the FD comps binding
                # shape-checks it against the ACA's capacity row count and
                # holds its pointer forever). Only the first
                # ``num_bands_now`` entries are maintained; tail slots carry
                # a valid placeholder (the last bound start) and are never
                # indexed — every kernel ``data_index`` is < num_bands_now.
                _full = self.xp.empty(self.alloc_capacity, dtype=self.xp.int32)
                _full[: _starts.shape[0]] = _starts
                if _starts.shape[0] < self.alloc_capacity:
                    _full[_starts.shape[0]:] = _starts[-1]
                self._min_freq_inds_store = _full
            else:
                self._min_freq_inds_store = _starts
            if self.use_template_arr:
                # The template twin shares the buffer's per-slot window
                # starts (same array object: in-place updates on cell swaps
                # reach both FD comps clones).
                self._acs_template_buffer.min_freq_inds = self._min_freq_inds_store

        # Routed: multi-shard buffers partition every engine call by owning
        # GPU split (and give each non-prototype device its own comp replica);
        # single-shard buffers pass straight through.
        self.rebuild_likelihood_engine()

        # TODO: fix this 4????
        self.special_band_inds = special_band_inds
        assert special_band_inds.shape[0] == self.params_interest.shape[0]
        self.now_index = self.get_index(special_band_inds)

    def rebuild_likelihood_engine(self):
        """(Re)build the routed band likelihood engine from CURRENT comps.

        Factored out of buffer construction so the in-run sig-het sweep
        (``GB_SIGHET_SWEEP``, gbspecialstretch) can swap ``gb_wdm_comp`` for
        a differently-configured engine instance and rebuild THIS -- the
        production wiring, byte-for-byte -- around it, then restore. Any
        cached per-device replicas die with the old router.
        """
        self._likelihood_engine = make_routed_band_engine(
            self._basis_settings,
            xp=self.xp,
            gb=self.gb,
            gb_fd_comp=self.gb_fd_comp,
            gb_wdm_comp=self.gb_wdm_comp,
            nchannels=self.nchannels,
            tdi_channel_setup=self.tdi_channel_setup,
            df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
            start_freq_inds=getattr(self, "start_freq_inds", None),
            data_length=self.data_length,
            opt_snr_rej_samp_limit=self.opt_snr_rej_samp_limit,
            snr_rej_detected=self.snr_rej_detected,
        )

    # ------------------------------------------------------------------
    # Views into the AnalysisContainerArray-backed scratch buffers
    # ------------------------------------------------------------------

    @property
    def acs_buffer(self) -> AnalysisContainerArray:
        """The :class:`AnalysisContainerArray` backing the per-band residual buffers.

        Post Buffer/ACA merge this *is* the buffer object itself; kept for
        back-compat with callers that used ``buffer.acs_buffer``.
        """
        return self

    # ------------------------------------------------------------------
    # Per-band buffer accessors
    # ------------------------------------------------------------------
    # In multi-GPU mode the buffer holds the bands sharded across GPUs
    # (striped by default; see ``_build_band_ac_list``). The shaped
    # accessors below return a :class:`BandView` that lets callers
    # index by global band number; reads/writes route to the owning
    # shard. In single-GPU mode they return the underlying ndarray
    # view directly (no overhead). The ``*_tmp`` flat accessors stay
    # single-GPU-only -- with multi-shard there is no single flat
    # ndarray, so callers should use the engine path (gb_likelihood
    # passes the list-of-shards through ``buffer_aca.linear_data_arr``
    # / ``linear_psd_arr`` directly).

    @property
    def _n_slots_alloc(self) -> int:
        """ALLOCATED slot count: ``alloc_capacity`` when fixed-capacity is
        active, else the bound count (allocation == binding, legacy)."""
        if getattr(self, "alloc_capacity", None) is not None:
            return int(self.alloc_capacity)
        return int(self.num_bands_now)

    def _shaped_or_view(self, acs, kind: str):
        """Return either the single-shard reshape (single-GPU) or a BandView (multi-GPU).

        Fixed capacity: the underlying ACA holds ``alloc_capacity`` slots
        but only the first ``num_bands_now`` are bound — expose exactly the
        bound FRONT (a leading-axis view single-shard; a bounded
        :class:`BandView` multi-shard). Without capacity this is verbatim
        the legacy accessor.
        """
        if len(acs.linear_data_arr) == 1:
            arr = acs.data_shaped[0] if kind == "data" else acs.psd_shaped[0]
            if (
                self.alloc_capacity is not None
                and int(arr.shape[0]) != int(self.num_bands_now)
            ):
                arr = arr[: int(self.num_bands_now)]
            return arr
        n_bands = (
            int(self.num_bands_now) if self.alloc_capacity is not None else None
        )
        return (
            acs.data_shaped_view(n_bands=n_bands)
            if kind == "data"
            else acs.psd_shaped_view(n_bands=n_bands)
        )

    def _flat_or_raise(self, acs, kind: str):
        if len(acs.linear_data_arr) == 1:
            return (
                acs.linear_data_arr[0] if kind == "data" else acs.linear_psd_arr[0]
            )
        raise RuntimeError(
            f"{kind}_buffer_tmp is only valid in single-GPU mode "
            "(multi-GPU buffers are a list of per-GPU shards). Use the "
            "engine path or BandView accessors instead."
        )

    @property
    def band_buffer_tmp(self):
        """Flat per-GPU residual buffer (1D view; single-GPU only)."""
        return self._flat_or_raise(self, "data")

    @property
    def band_buffer(self):
        """Per-band residual buffer indexable by global band id.

        Single-GPU: returns the ``(num_bands_now, nchannels, data_length)``
        reshape directly. Multi-GPU: returns a :class:`BandView` that
        routes per-band reads/writes through the owning shard.
        """
        return self._shaped_or_view(self, "data")

    @property
    def psd_buffer_tmp(self):
        """Flat per-GPU inverse-PSD buffer (1D view; single-GPU only)."""
        return self._flat_or_raise(self, "psd")

    @property
    def psd_buffer(self):
        """Per-band inverse-PSD buffer indexable by global band id.

        Same single-GPU / multi-GPU behaviour as :attr:`band_buffer`.
        """
        return self._shaped_or_view(self, "psd")

    @property
    def template_buffer_tmp(self):
        """Flat per-GPU template buffer (single-GPU only; ``use_template_arr`` True)."""
        return self._flat_or_raise(self._acs_template_buffer, "data")

    @property
    def template_buffer(self):
        """Per-band template buffer indexable by global band id.

        Same single-GPU / multi-GPU behaviour as :attr:`band_buffer`.
        """
        return self._shaped_or_view(self._acs_template_buffer, "data")

    # ------------------------------------------------------------------
    # Domain-aware allocation helpers
    # ------------------------------------------------------------------

    @property
    def basis_settings(self) -> DomainSettingsBase:
        """Parent basis-domain settings driving per-band buffer geometry."""
        return self._basis_settings

    # -- Slab-metadata cache (perf, 2026-08) ---------------------------
    # ``band_slab_Nf`` and ``slab_min_f`` are pure functions of the band
    # grid (``band_edges``, bind-constant), the parent basis settings, the
    # ``wdm_band_slab_layers`` knob (bind-constant) and the live per-slot
    # band binding (``unique_band_combos`` / ``num_bands_now``). They used
    # to be recomputed — with device syncs — on EVERY engine call (via
    # ``_slab_kernel_args`` / ``_adjust_via_engine`` /
    # ``refresh_row_metadata``). INVALIDATION CONTRACT: the binding only
    # changes through the ``special_indices_unique`` property setter and
    # ``resize_to`` — both call ``_invalidate_slab_metadata_cache`` — so the
    # cached values are computed once per bind and returned verbatim until
    # the next rebind. (Cache presence is tracked by attribute existence in
    # ``__dict__`` — no sentinel objects, so deepcopy/pickle stay safe.)

    def _invalidate_slab_metadata_cache(self) -> None:
        # ``_band_slab_Nf_cached`` is deliberately NOT dropped here (2026-08-28
        # shared-overhead audit). ``_compute_band_slab_Nf`` reads only
        # ``_wdm_band_slab_layers``, ``_basis_settings``, ``df`` and
        # ``band_edges`` -- all fixed at construction -- so the slab extent is
        # a RUN constant: a rebind changes which CELLS this buffer holds, not
        # the band grid. Dropping it made ``band_slab_Nf`` (the first reader
        # after every bind, via ``_get_fill_buffer_ind_map``) re-run
        # ``band_support_halfwidths``, a 1232-iteration python loop over
        # gbgpu's ``get_N``, ~2,100x/iteration -- 129.9 s/iteration (~6.6% of
        # a ~1954 s iteration), hidden inside the ``fill_indmap_data`` span.
        # The tell was a twin-call gap: ``fill_indmap_psd`` runs the same
        # function with the same inputs (only a bool differs) at 0.195
        # ms/chunk vs 56 ms for the data call. Keeping the value is
        # bit-identical; the two caches below DO consume the live binding and
        # must still go. See tests/test_band_slab_nf_cache.py.
        self.__dict__.pop("_slab_min_f_cached", None)
        # WDM per-slot start-layer store (see ``min_freq_inds``): constant
        # per bind (every entry is the parent ``ind_min_f``), invalidated on
        # the same rebind/resize hooks as the slab metadata.
        self.__dict__.pop("_min_freq_inds_wdm_cached", None)

    @property
    def band_slab_Nf(self) -> Optional[int]:
        """WDM-layer extent of a narrow per-band slab (task-b), or ``None``.

        ``None`` on the FD path, or on WDM when ``wdm_band_slab_layers`` is
        ``None`` (full-active-band layout). ``0`` auto-sizes to
        ``band_span + 2*(leakage + guard)``; ``N>0`` uses exactly ``N``. All
        slabs share this constant extent; each has its own origin in
        :attr:`slab_min_f`. Clamped to the parent active band.

        Cached per bind (see ``_invalidate_slab_metadata_cache``).
        """
        if "_band_slab_Nf_cached" in self.__dict__:
            return self._band_slab_Nf_cached
        # Dispatch through the class (not ``self``) so the property keeps
        # working on duck-typed stubs that borrow the raw ``fget``.
        out = SubBandBuffer._compute_band_slab_Nf(self)
        self._band_slab_Nf_cached = out
        return out

    def _compute_band_slab_Nf(self) -> Optional[int]:
        if self._wdm_band_slab_layers is None:
            return None
        if not isinstance(self._basis_settings, WDMSettings):
            return None
        if self._wdm_band_slab_layers > 0:
            slab_Nf = int(self._wdm_band_slab_layers)
            # Enforce the (previously comment-only, see ``slab_min_f``)
            # sizing contract ``slab_Nf >= max_band_span + 2*hw``: the
            # chunked-het kernels CLIP each source's per-layer window
            # ``m_floor +- m_band_half_width`` to the slab
            # (lat_chunked_het_kernels.hh, wdm_het_get_ll_impl) -- a slab
            # narrower than the widest band + the window spread silently
            # annihilates edge sources (d_h = h_h = 0, nothing subtracted).
            # This matters for variable-width band grids
            # (GB_BAND_EDGES_MODE=get_n), where a fixed
            # GB_WDM_BAND_SLAB_LAYERS tuned for 1-layer bands is too small
            # for the wide high-frequency bands. hw = 1 is the kernel-family
            # default (never overridden by gb_likelihood).
            _ldf = float(self.df) if not hasattr(self.df, "item") else self.df.item()
            _edges = self.xp.asarray(self.band_edges)
            # TOUCHED-LAYER span (band edges are free-floating frequencies,
            # 2026-08-15 user ruling -- never assume layer alignment): the
            # layers a band [lo, hi) touches are floor(lo/ldf) ..
            # ceil(hi/ldf) - 1, so a band straddling a layer boundary
            # counts BOTH layers. The 1e-6-layer epsilon keeps exactly
            # aligned edges (the uniform mode) from gaining a layer on
            # float noise -- bit-identical spans there.
            _lo = self.xp.floor(_edges[:-1] / _ldf + 1e-6).astype(int)
            _hi = self.xp.ceil(_edges[1:] / _ldf - 1e-6).astype(int)
            _max_span = max(1, int(self.xp.max(_hi - _lo)))
            _hw = 1
            if slab_Nf < _max_span + 2 * _hw:
                raise ValueError(
                    f"GB_WDM_BAND_SLAB_LAYERS={slab_Nf} is too small for "
                    f"this band grid: the widest band spans {_max_span} WDM "
                    f"layers and the per-source likelihood window adds "
                    f"m_band_half_width={_hw} on each side, so the slab "
                    f"must span >= {_max_span + 2 * _hw} layers (or use "
                    f"GB_WDM_BAND_SLAB_LAYERS=0 for auto-sizing). A "
                    f"too-small slab silently clips edge sources out of "
                    f"the likelihood."
                )
            # FD-support coverage check (2026-08-15 "strong checks" user
            # ruling): each band's slab must cover the maximum
            # single-source support of any source the band can hold,
            # i.e. [f_lo - get_N(f_hi)*df, f_hi + get_N(f_hi)*df]. With
            # the centered whole-layer origins below that means
            # slab_Nf >= touched_span + 2*ceil(support/layer_df). An
            # explicit slab below this floor raises loudly;
            # GB_SLAB_SUPPORT_CHECK=0 downgrades to a warning (escape
            # hatch for legacy grids where the +-1-layer WDM window
            # analysis, not the FD support envelope, is the operative
            # bound).
            _Tobs = float(getattr(self._basis_settings, "Tobs"))
            _support = band_support_halfwidths(
                np.asarray(asnumpy(self.band_edges), dtype=float), _Tobs
            )
            _sup_layers = np.ceil(_support / _ldf - 1e-6).astype(int)
            _spans = np.maximum(1, asnumpy(_hi) - asnumpy(_lo))
            _required = int(np.max(_spans + 2 * _sup_layers))
            if slab_Nf < min(_required, int(self._basis_settings.Nf_active)):
                _msg = (
                    f"GB_WDM_BAND_SLAB_LAYERS={slab_Nf} does not cover the "
                    f"worst-case single-source FD support of this band "
                    f"grid: a source at a band's top edge reaches "
                    f"get_N(f_hi)/Tobs = {float(_support.max()):.4e} Hz "
                    f"(= {int(_sup_layers.max())} layers) beyond the "
                    f"band, so the slab must span >= {_required} layers "
                    f"(touched span + 2*ceil(support/layer_df)). Use "
                    f"GB_WDM_BAND_SLAB_LAYERS=0 for auto-sizing, or set "
                    f"GB_SLAB_SUPPORT_CHECK=0 to accept the +-1-layer "
                    f"WDM-window bound instead."
                )
                if os.environ.get("GB_SLAB_SUPPORT_CHECK", "1") == "1":
                    raise ValueError(_msg)
                logger.warning(_msg)
        else:
            # Auto: cover the widest band + max(leakage, FD support) +
            # guard on each side (support-aware sizing, 2026-08-15
            # ruling; Tobs threads through so the slab always covers
            # [f_lo - get_N(f_hi)*df, f_hi + get_N(f_hi)*df]).
            # Class-qualified call (staticmethod): keeps the duck-typed
            # stub dispatch working (see the band_slab_Nf docstring).
            slab_Nf = SubBandBuffer.recommend_band_slab_layers(
                self.band_edges,
                float(self.df) if not hasattr(self.df, "item") else self.df.item(),
                leakage=_WDM_SLAB_LEAKAGE_LAYERS,
                guard=self._wdm_slab_guard_layers,
                xp=self.xp,
                Tobs=float(getattr(self._basis_settings, "Tobs")),
            )
        # Never exceed the parent active band. (If the support-aware
        # requirement exceeds Nf_active, the full active band is the
        # best possible coverage -- the legacy full-band semantics.)
        return int(min(slab_Nf, self._basis_settings.Nf_active))

    @staticmethod
    def recommend_band_slab_layers(band_edges, layer_df, leakage=_WDM_SLAB_LEAKAGE_LAYERS,
                                   guard=1, xp=np, Tobs=None, oversample=4,
                                   amp=1e-30) -> int:
        """Recommended per-band slab extent (WDM layers) for task-b.

        Band edges are FREE-FLOATING frequencies (2026-08-15 user ruling
        -- the scheduling division is independent of the WDM
        pixelization); only the STORAGE slabs are layer-derived, so each
        band's actual frequency range is floored/ceiled to whole layers
        here: the layers a band ``[lo, hi)`` TOUCHES are
        ``floor(lo/ldf) .. ceil(hi/ldf) - 1`` (a band straddling a layer
        boundary counts both layers; a 1e-6-layer epsilon keeps exactly
        aligned uniform-mode edges from gaining a layer on float noise,
        bit-identical there).

        Sizing = ``max over bands of (touched_span_b + 2 * (margin_b +
        guard))`` where ``margin_b = max(leakage,
        ceil(support_b / layer_df))``:

        * ``leakage`` (~2 layers for the recommended Tukey window)
          covers the WDM template spread beyond the carrier layers;
        * when ``Tobs`` is given, ``support_b = get_N(f_hi_b)/Tobs`` is
          the band's worst-case single-source FD half-support (leakage
          bookkeeping, 2026-08-15 "strong checks" ruling: the slab must
          cover ``[f_lo - get_N(f_hi)*df, f_hi + get_N(f_hi)*df]``).
          ``Tobs=None`` keeps the legacy leakage-only margins
          (back-compat for diagnostic scripts).

        This is what ``wdm_band_slab_layers=0`` (auto) resolves to;
        ``check_wdm_band_slab.py`` prints it alongside a measured
        leakage estimate.
        """
        edges = np.asarray(asnumpy(band_edges), dtype=float)
        ldf = float(layer_df)
        lo = np.floor(edges[:-1] / ldf + 1e-6).astype(int)
        hi = np.ceil(edges[1:] / ldf - 1e-6).astype(int)
        spans = np.maximum(1, hi - lo)
        margin = np.full(spans.shape, int(leakage))
        if Tobs is not None:
            support = band_support_halfwidths(
                edges, float(Tobs), oversample=oversample, amp=amp
            )
            margin = np.maximum(
                margin, np.ceil(support / ldf - 1e-6).astype(int)
            )
        return int(np.max(spans + 2 * (margin + int(guard))))

    @property
    def slab_min_f(self):
        """Per-slot start WDM layer of each narrow band slab (task-b), or ``None``.

        Each slot's slab spans ``[slab_min_f[slot], slab_min_f[slot] +
        band_slab_Nf)`` WDM layers, centered on the slot's band and clamped
        into the parent active band ``[ind_min_f, ind_max_f]``. Computed
        from the live per-slot band assignment (``unique_band_combos[:, 2]``)
        so it tracks cell swap-outs. Read by the chunked-het kernels (via
        ``fill_global_wdm`` / ``_slab_kernel_args``) as the per-slab layer
        origin. ``None`` when narrow slabs are off.

        Cached per bind: the band assignment only changes through the
        ``special_indices_unique`` setter / ``resize_to``, which invalidate
        the cache (see ``_invalidate_slab_metadata_cache``).
        """
        if "_slab_min_f_cached" in self.__dict__:
            return self._slab_min_f_cached
        # Dispatch through the class (not ``self``) so the property keeps
        # working on duck-typed stubs that borrow the raw ``fget``.
        out = SubBandBuffer._compute_slab_min_f(self)
        self._slab_min_f_cached = out
        return out

    def _compute_slab_min_f(self):
        slab_Nf = self.band_slab_Nf
        if slab_Nf is None:
            return None
        ldf = float(self.df) if not hasattr(self.df, "item") else self.df.item()
        band_inds = self.unique_band_combos[:, 2]
        # Center the slab on the band's TOUCHED whole-layer range (band
        # edges are free-floating frequencies, 2026-08-15 user ruling --
        # never assume layer alignment): the layers a band [lo, hi)
        # touches are floor(lo/ldf) .. ceil(hi/ldf) - 1, so a band
        # straddling a layer boundary is centered over BOTH layers. The
        # 1e-6-layer epsilon keeps exactly aligned uniform-mode edges
        # bit-identical (center formula reduces to the legacy
        # (lo + hi) // 2 there). Coverage: slab_Nf (support-aware, see
        # _compute_band_slab_Nf) >= touched_span + margins each side.
        lo_layer = self.xp.floor(
            self.band_edges[band_inds] / ldf + 1e-6
        ).astype(self.xp.int32)
        hi_layer_excl = self.xp.ceil(
            self.band_edges[band_inds + 1] / ldf - 1e-6
        ).astype(self.xp.int32)
        center = (lo_layer + hi_layer_excl - 1) // 2
        origins = center - slab_Nf // 2
        parent = self._basis_settings
        lo = int(parent.ind_min_f)
        hi = max(lo, int(parent.ind_max_f) + 1 - slab_Nf)
        out = self.xp.clip(origins, lo, hi).astype(self.xp.int32)
        n_alloc = self._n_slots_alloc
        if int(out.shape[0]) < n_alloc:
            # Fixed capacity: pad to the ALLOCATED slot count so per-slot
            # kernel metadata and the shard views (which row-slice by the
            # capacity-sized gpu_splits) stay shape-consistent. Tail values
            # (valid clamp floor) are never consumed — every kernel
            # data_index is < num_bands_now.
            out = self.xp.concatenate([
                out,
                self.xp.full(n_alloc - int(out.shape[0]), lo, dtype=self.xp.int32),
            ])
        return out

    @property
    def _per_band_data_shape(self) -> tuple:
        """Shape of a single band's residual buffer (one AC's data_res_arr)."""
        if isinstance(self._basis_settings, FDSettings):
            return (self.nchannels, self._fd_store_length)
        elif isinstance(self._basis_settings, WDMSettings):
            # Task-b: a narrow per-band slab covers ``band_slab_Nf`` layers
            # (centered on the band, origin in ``slab_min_f``) instead of the
            # full active grid; ``band_slab_Nf is None`` keeps the full
            # ``Nf_active`` layout (the WDM kernel then uses the single global
            # [ind_min_f, ind_max_f] origin).
            Nf_use = self._per_band_Nf
            Nt_active = self._basis_settings.Nt_active
            return (self.nchannels, Nf_use, Nt_active)
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_Nf(self) -> int:
        """Per-band WDM slab frequency extent: ``band_slab_Nf`` when narrow,
        else the full parent ``Nf_active``."""
        slab_Nf = self.band_slab_Nf
        return int(slab_Nf) if slab_Nf is not None else int(self._basis_settings.Nf_active)

    @property
    def _per_band_sens_shape(self) -> tuple:
        """Shape of a single band's inverse-PSD buffer (one AC's sens_mat.invC)."""
        if isinstance(self._basis_settings, FDSettings):
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, self._fd_store_length)
            return (self.nchannels, self._fd_store_length)
        elif isinstance(self._basis_settings, WDMSettings):
            Nf_use = self._per_band_Nf
            Nt_active = self._basis_settings.Nt_active
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, Nf_use, Nt_active)
            return (self.nchannels, Nf_use, Nt_active)
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_data_dtype(self):
        """Element dtype for the per-band residual buffer."""
        if isinstance(self._basis_settings, FDSettings):
            return self.xp.complex128
        elif isinstance(self._basis_settings, WDMSettings):
            return self.xp.float64
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_sens_dtype(self):
        """Element dtype for the per-band inverse-PSD buffer.

        FD is REAL: the gb_fd kernels consume the real part of the inverse
        covariance (Hermitian; the imaginary parts cancel in the quadratic
        forms), matching the FDDomain double* layout.
        """
        if isinstance(self._basis_settings, FDSettings):
            return self.xp.float64
        elif isinstance(self._basis_settings, WDMSettings):
            return self.xp.float64
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    def _build_per_band_basis_settings(self) -> DomainSettingsBase:
        """Construct the per-band domain settings used by each per-band AC.

        Each per-band AC's data domain needs a settings object whose
        ``basis_shape_active`` matches the per-band data shape. For FD this
        is a fresh FDSettings sized to the FD store length. For WDM the
        per-band settings share the parent's active grid.
        """
        if isinstance(self._basis_settings, FDSettings):
            return FDSettings(
                N=self._fd_store_length,
                df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
                force_backend=self._basis_settings.backend_name.split("_", 1)[1],
            )
        elif isinstance(self._basis_settings, WDMSettings):
            # Per-band WDMSettings: full parent active band by default. Task-b:
            # when narrow slabs are on, the per-band domain only needs the
            # SHAPE (band_slab_Nf, Nt_active) -- the actual per-slot layer
            # origin lives in ``slab_min_f`` (the kernel arg), not here -- so
            # we pin ``ind_max_f`` to make ``Nf_active == band_slab_Nf``.
            parent = self._basis_settings
            # The slab extent must ride IN the constructor args: consumers
            # (WDMDomain.__init__) re-build the settings from args/kwargs, so
            # a post-construction ``ind_max_f`` mutation is silently lost and
            # the per-band domain reverts to the full active band (shape
            # assert). ``ind_max_f = int(max_freq / layer_df)`` (inclusive);
            # the half-layer offset keeps the floor rounding-safe.
            slab_Nf = self.band_slab_Nf
            _ind_max_f = (
                int(parent.ind_min_f) + int(slab_Nf) - 1
                if slab_Nf is not None
                else int(parent.ind_max_f)
            )
            per_band = WDMSettings(
                Nf=parent.Nf,
                Nt=parent.Nt,
                dt=parent.data_dt,
                t0=parent.t0,
                oversample=parent.oversample,
                window=parent.window,
                omega=parent.omega,
                min_freq=parent.ind_min_f * parent.layer_df,
                max_freq=(_ind_max_f + 0.5) * parent.layer_df,
                min_time=parent.ind_min_t * parent.layer_dt,
                max_time=parent.ind_max_t * parent.layer_dt,
            )
            return per_band
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    def _build_band_ac_list(self) -> tuple:
        """Allocate one :class:`AnalysisContainer` per active cell.

        Returns ``(ac_list, aca_kwargs)`` ready to be fed into
        ``AnalysisContainerArray.__init__`` (either on ``self`` or on the
        template twin). Branches on the parent basis domain: FD buffers are
        complex with ``complex_psd=True`` (XYZ CSD support); WDM buffers are
        real-valued slabs over the full active grid.
        """
        per_band_settings = self._build_per_band_basis_settings()
        data_shape = self._per_band_data_shape
        sens_shape = self._per_band_sens_shape
        data_dtype = self._per_band_data_dtype
        sens_dtype = self._per_band_sens_dtype

        # Forward sensitivity matrix: full per-band allocation only when the
        # caller asks to keep it (``keep_sens_mat=True``). Otherwise back it
        # with a zero-storage broadcast view -- correct ``.shape`` / ``.ndim``
        # (the ACA reads ``sens_mat.shape`` to derive ``shape_sens``; the
        # SensitivityMatrixBase ndarray setter only shape-checks it) but a
        # single element of memory. ``invC`` -- the only array the likelihood
        # kernels consume -- is always a real per-band buffer.
        def _forward_sens():
            if self.keep_sens_mat:
                return cp.zeros(sens_shape, dtype=sens_dtype)
            return cp.broadcast_to(cp.zeros((), dtype=sens_dtype), sens_shape)

        # Fixed capacity: allocate EVERY per-slot container at
        # ``alloc_capacity`` (the bound count ``num_bands_now`` only limits
        # the front views); default path allocates exactly the bound count.
        n_alloc = self._n_slots_alloc
        ac_list = []
        for _ in range(n_alloc):
            res_data = cp.zeros(data_shape, dtype=data_dtype)
            data_domain = per_band_settings.associated_class(res_data, per_band_settings)
            sm = SensitivityMatrixBase(per_band_settings, skip_inv_det=True)
            sm.sens_mat = _forward_sens()
            sm.invC = cp.zeros(sens_shape, dtype=sens_dtype)
            sm.channel_shape = sens_shape[: -len(per_band_settings.basis_shape_active)]
            ac_list.append(AnalysisContainer(data_domain, sm))

        gpus_in = getattr(self.gb, "gpus", None) if self.backend.uses_cupy else None
        # Multi-GPU at the GB band-tree level (parallel-resources plan P1):
        # group the per-cell slabs by their BAND id so every cell of a band
        # -- in particular a tempering swap pair (same band, adjacent temps)
        # -- is device-local, while bands round-robin across GPUs in
        # first-appearance order (the GB move activates bands in even/odd
        # parity rotation, so each parity pass still spreads over all
        # devices). Plain slot striping put a row's temps in consecutive
        # slots on ALTERNATING shards, making every tempering swap pair pay
        # a cross-device hop. The per-band accessors (band_buffer /
        # psd_buffer / template_buffer) automatically fall back to a single
        # ndarray view for single-GPU runs and return a BandView
        # (multi-shard router) otherwise -- see the accessor block above.
        _group_ids = asnumpy(self.unique_band_combos[:, 2])
        if len(ac_list) > _group_ids.shape[0]:
            # Fixed capacity: tail (unbound) slots need group ids too. Give
            # each its own fresh pseudo-band id so band_gpu_assignment
            # round-robins them across devices — keeping every shard's
            # bound-slot count balanced after any resize_to(k). Grouping is
            # a device-locality optimization only; routing correctness never
            # depends on it (BandView / _RoutedBandEngine route per split).
            _group_ids = np.concatenate([
                np.asarray(_group_ids, dtype=int),
                int(np.max(_group_ids)) + 1
                + np.arange(len(ac_list) - _group_ids.shape[0], dtype=int),
            ])
        gpu_assignment = (
            band_gpu_assignment(
                len(ac_list),
                list(gpus_in),
                group_ids=_group_ids,
            )
            if gpus_in
            else None
        )
        aca_kwargs = dict(
            gpus=list(gpus_in) if gpus_in else None,
            complex_psd=False,
            gpu_assignment=gpu_assignment,
        )
        return ac_list, aca_kwargs

    def update_special_indices(self, new_special_indices, inds_fill=None):
        if inds_fill is None:
            inds_fill = cp.arange(self.num_bands_now)

        assert inds_fill.shape[0] == new_special_indices.shape[0]
        if getattr(self, "_specials_placeholder", False):
            # resize_to contract: the first update after a resize MUST be a
            # FULL rebind (inds_fill covering every bound slot) — a partial
            # fill would mix the fresh cells with the resize placeholders.
            assert int(inds_fill.shape[0]) == int(self.num_bands_now), (
                "SubBandBuffer.update_special_indices after resize_to() must "
                "be a FULL rebind (inds_fill spanning all num_bands_now "
                f"slots); got {int(inds_fill.shape[0])} of "
                f"{int(self.num_bands_now)}."
            )
            self._specials_placeholder = False
        _tmp_indices = self.special_indices_unique.copy()
        _tmp_indices[inds_fill] = new_special_indices
        # The property setter below rebuilds the sort table, band combos,
        # window starts and frequency limits FROM this (num_bands_now,)-long
        # array — after a full rebind no capacity-sized or previous-binding
        # entry survives anywhere ``get_index`` can see.
        self.special_indices_unique = _tmp_indices

    def resize_to(self, k: int) -> None:
        """Rebind this fixed-capacity buffer to ``k`` slots (front of the alloc).

        Re-slices every bound-count-dependent view to the first ``k`` slots
        of the capacity allocation and sets ``num_bands_now = k``. NO data
        is touched: the slot slabs, the PSD slabs, the template twin and the
        FD window-start store keep their memory (and the engine bindings
        their pointers) — the follow-up rebind refills them.

        CONTRACT (stale-specials hazard): the special-indices map is FULLY
        INVALID after this call — the entries are placeholders sized to
        ``k``, NOT live cells. The caller MUST immediately follow with a
        FULL rebind: ``update_special_indices(new_specials,
        inds_fill=xp.arange(k))`` (BandSorter.get_buffer's resize-rebind
        path does exactly this), whose property setter rebuilds
        ``special_indices_unique`` / ``special_indices_unique_sort`` /
        ``unique_band_combos`` / window starts at length ``k`` from the new
        specials — leaving no stale entry from any previous (possibly
        larger) binding visible to :meth:`get_index`. Until then
        :meth:`get_index` hard-errors and partial
        :meth:`update_special_indices` fills are rejected.
        """
        assert self.alloc_capacity is not None, (
            "resize_to requires a fixed-capacity buffer (alloc_capacity set "
            "at construction)."
        )
        k = int(k)
        assert 0 < k <= int(self.alloc_capacity), (
            f"resize_to({k}) outside (0, alloc_capacity="
            f"{int(self.alloc_capacity)}]."
        )
        if k == int(self.num_bands_now):
            return
        # The bound-slot count changes below (before the placeholder rebind
        # runs the specials setter): drop the cached slab metadata now.
        self._invalidate_slab_metadata_cache()
        xp = get_array_module(self._special_indices_unique)
        old = self._special_indices_unique
        if int(old.shape[0]) >= k:
            placeholder = old[:k].copy()
        else:
            # Growing: pad with the last live special — VALID values only
            # (the specials setter unpacks them into band indices), never
            # consumed (the mandatory full rebind overwrites every entry).
            placeholder = xp.concatenate(
                [old, xp.repeat(old[-1:], k - int(old.shape[0]))]
            )
        self.num_bands_now = k
        self.psd_shape = (k,) + self._per_band_sens_shape
        # Run the FULL property setter on the placeholder so every derived
        # per-slot array (sort table, band combos, window starts, frequency
        # limits) is length-k and internally consistent — nothing is left
        # at the previous binding's length. The arrays still describe
        # placeholder cells, hence the flag below.
        self.special_indices_unique = placeholder
        self._specials_placeholder = True

    @property
    def special_indices_unique(self):
        return self._special_indices_unique

    @special_indices_unique.setter
    def special_indices_unique(self, special_indices_unique):
        # Rebind: the per-slot band assignment changes here, so the cached
        # slab metadata (band_slab_Nf / slab_min_f) must be recomputed on
        # next read (see _invalidate_slab_metadata_cache).
        self._invalidate_slab_metadata_cache()
        self._special_indices_unique_sort = self.xp.argsort(special_indices_unique)
        self._special_indices_unique = special_indices_unique

        _temp_inds, _walker_inds, _band_inds = self.get_separate_inds_from_special_index(
            special_indices_unique
        )

        self.unique_band_combos = self.xp.array([_temp_inds, _walker_inds, _band_inds]).T

        if self.num_bands == 1:
            tmp_buffer_start_index = (self.band_edges[0] / self.df).astype(
                np.int32
            ) - self.edge_buffer
            if getattr(self, "_parent_ind_min", None) is None:
                # Legacy full-grid parent: the single window must cover the
                # whole band plus both edge buffers. (With a frequency-
                # clipped parent this is un-satisfiable by construction --
                # the clamp below shifts/shrinks the window instead.)
                assert tmp_buffer_start_index + self._fd_store_length >= (
                    (self.band_edges[-1] / self.df).astype(np.int32) + self.edge_buffer
                )
            self.buffer_start_index = self.xp.repeat(
                tmp_buffer_start_index, self.unique_band_combos.shape[0]
            )

        else:
            self.buffer_start_index = (
                self.band_edges[self.unique_band_combos[:, 2] - 1] / self.df
            ).astype(np.int32)
            self.buffer_start_index[self.unique_band_combos[:, 2] == 0] = (
                self.band_edges[0] / self.df
            ).astype(np.int32) - self.edge_buffer
            # Clamp so buffer end never overflows the data range (band_edges[-1])
            max_start = int(self.band_edges[-1] / self.df) - self._fd_store_length
            self.buffer_start_index = np.minimum(self.buffer_start_index, max_start)

        if getattr(self, "_parent_ind_min", None) is not None:
            # Frequency-clipped parent: clamp every window into the stored
            # bin range [ind_min, ind_min + stored_len - window]. Edge cells
            # lose their out-of-domain guard margin (the parent stores
            # nothing there); the engines read placement from
            # ``start_freq_inds`` so a shifted window stays consistent.
            lo = self._parent_ind_min
            hi = max(lo, lo + self._parent_stored_len - self._fd_store_length)
            self.buffer_start_index = self.xp.clip(self.buffer_start_index, lo, hi)

        self.start_freq_inds = self.xp.asarray(self.buffer_start_index.copy().astype(np.int32))
        if hasattr(self, "_min_freq_inds_store"):
            # in-place: the FD comps clone holds a pointer to this array.
            # Front-write only: on a fixed-capacity buffer the store is
            # capacity-sized while ``start_freq_inds`` covers the bound
            # slots; without capacity the two lengths are equal (identical
            # to the old full-slice write).
            self._min_freq_inds_store[
                : self.start_freq_inds.shape[0]
            ] = self.start_freq_inds

        # Band window of every slot. On RJ-provenance buffers this is
        # widened by N/4 "to allow to move over band edge when proposing
        # in-model" -- births themselves need no widening at all, since a
        # dead row's band is DERIVED from its own drawn f0 (the
        # ``band_inds = searchsorted(band_edges, freqs)`` below), so a
        # birth is inside its band by construction and the RJ support
        # gate is a tautology for it.
        #
        # The widening's units are wrong by ``layer_df * Tobs == Nt/2``
        # (1080x on the v7 3-month grid) -- see :func:`rj_band_window`.
        # GB_BAND_WINDOW_STRICT=1 drops it, leaving the in-model gate's
        # own ``n4_s`` (correctly in move-df bins) as the ONLY N/4.
        self.frequency_lims = rj_band_window(
            self.band_edges,
            self.band_N_vals,
            self.unique_band_combos[:, 2],
            self.df,
            self.is_rj,
        )

    @property
    def _fd_store_length(self) -> int:
        """FD frequency-bin count of one cell's residual window.

        This is the FD-specific store size handed in at construction. It is
        deliberately kept separate from the inherited ACA ``data_length``
        (which on the WDM path becomes ``Nf_active * Nt_active`` -- the
        linear-buffer stride the inherited packing methods need).
        """
        return self._fd_store_length_value

    @property
    def min_freq_inds(self):
        """Per-slot minimum-frequency index, unified across domains.

        FD: absolute FD bin where each cell's residual window starts
        (``buffer_start_index``). WDM: the start layer of each cell's slab --
        currently the parent grid's ``ind_min_f`` for every slot, until the
        WDM kernel takes per-band layer offsets. The likelihood engines read
        this off the buffer at every call, so it never goes stale across
        band swap-outs.
        """
        if isinstance(self._basis_settings, WDMSettings):
            # ALLOCATED length (== bound length without fixed capacity): the
            # comps bindings and shard views shape-check / row-slice this
            # against the ACA's allocated row count. Cached per bind (perf,
            # 2026-08): this used to allocate a fresh device array on EVERY
            # engine call; the value is bind-constant (parent ``ind_min_f``
            # repeated), so compute once and invalidate on the same
            # rebind/resize hooks as the slab metadata
            # (``_invalidate_slab_metadata_cache``). Returning a persistent
            # array is also strictly more pointer-stable for any binding
            # that caches it.
            cached = self.__dict__.get("_min_freq_inds_wdm_cached")
            n_alloc = int(self._n_slots_alloc)
            if cached is None or int(cached.shape[0]) != n_alloc:
                cached = self.xp.full(
                    n_alloc, self._basis_settings.ind_min_f, dtype=self.xp.int32
                )
                self._min_freq_inds_wdm_cached = cached
            return cached
        return self._min_freq_inds_store

    @property
    def special_indices_unique_sort(self):
        return self._special_indices_unique_sort

    @staticmethod
    def _materialize(buf):
        """Return a single ndarray view for ``buf``.

        Single-shard runs already expose ``buf`` as a reshape view of the
        underlying ndarray, so this is a no-op (returns ``buf`` itself).
        Multi-shard runs expose ``buf`` as a :class:`BandView` -- gather it
        to a single ndarray on ``gpus[0]`` so downstream einsum / boolean
        indexing / sum kernels see a contiguous array.
        """
        if isinstance(buf, BandView):
            return buf.gather()
        return buf

    def likelihood(
        self, source_only: bool = False, noise_only: bool = False, slots=None
    ) -> float:
        """Band-level log-likelihood over all cells in the buffer.

        Overrides the inherited per-AC ``AnalysisContainerArray.likelihood``
        dispatch: the buffer computes its cell likelihoods directly from the
        shaped residual / PSD views (vectorized over cells).

        ``slots`` (optional, ``source_only`` ONLY): score just these slot
        indices and return the per-slot values IN THE GIVEN ORDER (length
        ``len(slots)``), instead of the whole buffer. Every cell's value is
        an independent per-row reduction of its own slab
        (``_reduce`` contracts each row separately), so a subset carries
        exactly the numbers the full call would place at those positions --
        the subset only removes work. Used by the tempering stage, which
        needs two columns of a temperature pair, not the whole ladder
        (``run_tempering``'s "add indices because not every likelihood is
        needed" TODO), and skips cells that hold no sources at all.
        """
        assert not (source_only and noise_only)
        if slots is not None and not source_only:
            raise ValueError(
                "SubBandBuffer.likelihood(slots=...) requires source_only=True: "
                "the PSD log-determinant term is a whole-buffer sum and has no "
                "per-slot restriction."
            )

        # band_buffer / template_buffer / psd_buffer are either ndarrays
        # (single-GPU; in-place mutation rolls back into the underlying
        # buffer) or BandView (multi-GPU; mutating after materialisation
        # PER-SHARD, CHUNKED reduction (2026-08-14). The previous pipeline
        # gathered the FULL band-sharded buffer to gpus[0], copied it for
        # the template subtraction, and let einsum transpose-copy it again:
        # transients scaling with TOTAL slots (~50 GB at 32,768 slots, all
        # on one device) that OOM'd job 180 the moment concurrent lanes
        # made them coexist. Each shard now reduces on its OWNING device in
        # GB_BAND_LL_CHUNK-slot chunks; only the per-band scalars move.
        # Domain-generic inner product: <a|b> = 4 sum(a* invC b) * dc (dc =
        # basis measure; FD df, WDM pixel measure), trailing axes flattened
        # -- identical arithmetic per band, chunking only batches it.
        if noise_only and self.tdi_channel_setup == "XYZ":
            raise NotImplementedError(
                "Noise-only likelihood requires log-determinant over "
                "frequency for XYZ CSD.")

        nc = self.nchannels
        dc = float(self.settings.differential_component)
        chunk = max(1, int(os.environ.get("GB_BAND_LL_CHUNK", "1024")))
        xyz = self.tdi_channel_setup == "XYZ"

        def _reduce(num, psd):
            k = num.shape[0]
            nf = num.reshape(k, nc, -1)
            if xyz:
                pf = psd.reshape(k, nc, nc, -1)
                return (-(1.0 / 2.0) * 4.0 * dc * cp.einsum(
                    "bik,bijk,bjk->b", nf.conj(), pf, nf).real)
            pf = psd.reshape(k, nc, -1)
            return (-(1.0 / 2.0) * 4.0 * dc
                    * cp.sum((nf.conj() * nf) * pf, axis=(1, 2)).real)

        band = self.band_buffer
        psd_b = self.psd_buffer
        tmpl = self.template_buffer if self.use_template_arr else None

        if slots is not None:
            return self._likelihood_slots(slots, band, psd_b, tmpl, _reduce, chunk)

        if isinstance(band, BandView):
            aca = band._aca
            nb_tot = int(band.shape[0])
            out_host = np.empty(nb_tot, dtype=np.float64)
            psd_log_acc = 0.0
            uses_dev = getattr(aca, "gpus", None) is not None
            main_dev = cp.cuda.runtime.getDevice() if uses_dev else None
            # Hoist the ``_shards`` property evaluations out of the loop:
            # each access rebuilds the per-shard view list (and re-enters
            # device contexts per shard on the real ACA).
            band_shards = band._shards
            psd_shards = psd_b._shards
            tmpl_shards = tmpl._shards if tmpl is not None else None
            try:
                for s in range(len(band_shards)):
                    ids = np.asarray(asnumpy(aca.gpu_splits[s]), dtype=int)
                    if uses_dev:
                        cp.cuda.runtime.setDevice(int(aca.gpus[s]))
                    d_sh = band_shards[s]
                    p_sh = psd_shards[s]
                    t_sh = tmpl_shards[s] if tmpl_shards is not None else None
                    for c0 in range(0, d_sh.shape[0], chunk):
                        c1 = min(c0 + chunk, d_sh.shape[0])
                        num = (d_sh[c0:c1] - t_sh[c0:c1]
                               if t_sh is not None else d_sh[c0:c1])
                        if not noise_only:
                            out_host[ids[c0:c1]] = asnumpy(
                                _reduce(num, p_sh[c0:c1]))
                        if noise_only or not source_only:
                            pc = p_sh[c0:c1]
                            nz = pc[pc != 0.0]
                            psd_log_acc += float(asnumpy(-cp.sum(cp.log(
                                cp.abs(1 / nz if noise_only else nz)))))
            finally:
                if uses_dev:
                    cp.cuda.runtime.setDevice(main_dev)
            if noise_only:
                return psd_log_acc
            source_term = cp.asarray(out_host)
        else:
            nb_tot = int(band.shape[0])
            source_term = cp.empty(nb_tot, dtype=cp.float64)
            psd_log_acc = 0.0
            for c0 in range(0, nb_tot, chunk):
                c1 = min(c0 + chunk, nb_tot)
                num = (band[c0:c1] - tmpl[c0:c1]
                       if tmpl is not None else band[c0:c1])
                if not noise_only:
                    source_term[c0:c1] = _reduce(num, psd_b[c0:c1])
                if noise_only or not source_only:
                    pc = psd_b[c0:c1]
                    nz = pc[pc != 0.0]
                    psd_log_acc += float(asnumpy(-cp.sum(cp.log(
                        cp.abs(1 / nz if noise_only else nz)))))
            if noise_only:
                return psd_log_acc

        if source_only:
            return source_term

        # Diagonal noise_term fall_back # TODO check if this is sufficient not used currently anyway
        if self.tdi_channel_setup == "XYZ":
            warnings.warn("The current psd ll calculation is not correct for XYZ CSD channel setup.")

        return source_term + psd_log_acc

    def _likelihood_slots(self, slots, band, psd_b, tmpl, _reduce, chunk):
        """``likelihood(source_only=True, slots=...)`` -- the subset path.

        Gathers only the requested slot rows (bounded at ``chunk`` rows per
        pass, same transient budget as the full path) and reduces them with
        the SAME per-row ``_reduce`` kernel, so each returned value is
        bit-for-bit what the whole-buffer call produces for that slot.
        Multi-GPU: slots are routed to their owning shard through
        ``gpu_splits`` and reduced on that shard's device -- no extra cross
        device traffic, and the shard/split bookkeeping is read-only here.
        """
        n_sl = int(slots.shape[0])
        if n_sl == 0:
            return self.xp.zeros(0, dtype=cp.float64)

        if isinstance(band, BandView):
            aca = band._aca
            nb_tot = int(band.shape[0])
            sl_h = np.asarray(asnumpy(slots), dtype=np.int64)
            out_host = np.empty(n_sl, dtype=np.float64)
            uses_dev = getattr(aca, "gpus", None) is not None
            main_dev = cp.cuda.runtime.getDevice() if uses_dev else None
            band_shards = band._shards
            psd_shards = psd_b._shards
            tmpl_shards = tmpl._shards if tmpl is not None else None
            # global slot -> (owning shard, row inside that shard): the
            # CACHED static-layout maps (2026-08-27 tempering audit -- this
            # rebuilt the host maps + asnumpy'd every gpu_split on EVERY
            # per-pair scoring call, ~40k times/iteration).
            owner, local = shard_lookup_maps(aca)
            owner_of = owner[sl_h]
            try:
                for s in range(len(band_shards)):
                    pos = np.where(owner_of == s)[0]
                    if pos.shape[0] == 0:
                        continue
                    if uses_dev:
                        cp.cuda.runtime.setDevice(int(aca.gpus[s]))
                    rows = cp.asarray(local[sl_h[pos]])
                    d_sh = band_shards[s]
                    p_sh = psd_shards[s]
                    t_sh = tmpl_shards[s] if tmpl_shards is not None else None
                    for c0 in range(0, pos.shape[0], chunk):
                        c1 = min(c0 + chunk, pos.shape[0])
                        r = rows[c0:c1]
                        num = (d_sh[r] - t_sh[r]) if t_sh is not None else d_sh[r]
                        out_host[pos[c0:c1]] = asnumpy(_reduce(num, p_sh[r]))
            finally:
                if uses_dev:
                    cp.cuda.runtime.setDevice(main_dev)
            return cp.asarray(out_host)

        sl = self.xp.asarray(slots)
        source_term = self.xp.empty(n_sl, dtype=cp.float64)
        for c0 in range(0, n_sl, chunk):
            c1 = min(c0 + chunk, n_sl)
            idx = sl[c0:c1]
            num = (band[idx] - tmpl[idx]) if tmpl is not None else band[idx]
            source_term[c0:c1] = _reduce(num, psd_b[idx])
        return source_term

    # Explicit alias while callers migrate off the ``likelihood`` name (which
    # shadows the inherited per-AC ACA dispatch).
    band_likelihoods = likelihood

    def _to_phys(self, params, leaf_inds=None):
        """Sampling -> physical rows through the transform container.

        ``leaf_inds`` (per-row leaf indices) is required by containers built
        with a per-leaf ``fill_dict`` list (Eryn validates); scalar-fill
        containers ignore it.
        """
        return self.transform_fn.both_transforms(params, xp=cp, leaf_inds=leaf_inds)

    def get_swap_ll(self, params_remove, params_add, data_index, N_vals, phase_maximize=False,
                    leaf_inds=None):
        """Per-proposal swap log-likelihood difference.

        Domain-agnostic: dispatches to ``self._likelihood_engine.get_swap_ll``,
        which is either :class:`FDBandLikelihoodEngine` or
        :class:`WDMBandLikelihoodEngine` depending on the buffer's
        ``basis_settings``. Both engines take the per-band ACA (``self``)
        and the physical params, and return a :class:`SwapLLResult`. The
        rejection-sampling clamp and the phase-maximisation correction live
        here so the engine stays a thin wrapper around the kernel.
        """
        params_remove_phys = self._to_phys(params_remove, leaf_inds=leaf_inds)
        params_add_phys = self._to_phys(params_add, leaf_inds=leaf_inds)

        result = self._likelihood_engine.get_swap_ll(
            self,
            params_remove_phys,
            params_add_phys,
            data_index=data_index,
            noise_index=data_index,
            N_vals=N_vals,
            phase_maximize=phase_maximize,
            waveform_kwargs=self.waveform_kwargs,
        )

        ll_diff = result.ll_diff
        kept = result.kept

        if np.any(~kept):
            logger.info(f"NOT KEEPING: {(~kept).sum()}")

        if phase_maximize and result.phase_angle is not None:
            # Engine returns the per-proposal phase rotation applied during
            # phase-maximisation; subtract it from phi0 so the accepted
            # parameters reflect the maximised draw.
            params_add[kept, self._phi0_col] = (
                params_add[kept, self._phi0_col] - result.phase_angle
            )

        # Rejection sampling on SNR: only applied to *add* proposals (the
        # remove side's opt_snr is meaningless when amp_add is tiny).
        reject = self.xp.zeros(kept.shape[0], dtype=bool)
        _bad_swap = result.opt_snr_add[kept] < self.opt_snr_rej_samp_limit
        if getattr(self, "snr_rej_detected", False) and (
                getattr(result, "d_h_add", None) is not None):
            _det_add = self.xp.asarray(result.d_h_add)[kept].real / self.xp.maximum(
                result.opt_snr_add[kept], 1e-300)
            _bad_swap = _bad_swap | (_det_add < self.opt_snr_rej_samp_limit)
        reject[kept] = _bad_swap & (params_add_phys[kept, 0] > 1e-30)
        ll_diff[reject] = -1e300

        return ll_diff

    def get_ll(self, params, data_index, noise_index, N_vals, phase_maximize=False,
               return_inner_products=False, leaf_inds=None):
        """Per-source log-likelihood against the cell residuals.

        Domain-agnostic dispatch like :meth:`get_swap_ll`. Returns the
        log-likelihood array ``-0.5 * (d_d + h_h - 2 d_h)`` on the engine's
        xp module (``d_d`` per the underlying computation object's
        convention; 0 unless configured). With
        ``return_inner_products=True`` returns ``(ll, d_h, h_h,
        phase_angle)`` instead. The raw inner products also land on
        :attr:`d_h_out` / :attr:`h_h_out`, and :attr:`phase_angle` carries
        the maximising rotation when ``phase_maximize=True``.
        """
        # Speed-diagnosis spans (user directive 2026-08-15: production runs
        # 20-100x slower than the kernel benches; decompose every hot call).
        _dtm = getattr(self, "_prop_timer", None)
        if _dtm is not None:
            _dtm.count("gll_calls")
            _dtm.count("gll_rows", int(params.shape[0]))
        with _tspan(_dtm, "gll_to_phys"):
            params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        with _tspan(_dtm, "gll_engine"):
            ll = self._likelihood_engine.get_ll(
                self,
                params_phys,
                data_index=data_index,
                noise_index=noise_index,
                N_vals=N_vals,
                phase_maximize=phase_maximize,
                waveform_kwargs=self.waveform_kwargs,
            )
        self.d_h_out = self._likelihood_engine.d_h_out
        self.h_h_out = self._likelihood_engine.h_h_out
        self.phase_angle = self._likelihood_engine.phase_angle
        self.kept_out = getattr(
            self._likelihood_engine, "kept_out",
            self.xp.ones(params.shape[0], dtype=bool),
        )
        if return_inner_products:
            return ll, self.d_h_out, self.h_h_out, self.phase_angle
        return ll

    def setup_in_model_likelihood(self, params, data_index, N_vals=None, leaf_inds=None) -> None:
        """Per-source in-model likelihood setup (once per repeat block).

        Forwards the picked sources' CURRENT sampling-basis params
        (transformed to physical) plus their buffer slots to the engine's
        ``setup_in_model`` hook. Chunked-het / FD engines no-op; a sig-het
        computation builds its heterodyne reference against the
        source-free cell residuals here and holds it constant until
        :meth:`clear_in_model_likelihood`. Call AFTER the sources are
        removed from the residual, BEFORE the reference ll of the repeat
        block is computed.

        Returns the engine hook's value: truthy when a sig-het reference
        is now active (the move uses this to arm its mid-block drift
        refresh), ``None`` from the no-op hooks."""
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        return self._likelihood_engine.setup_in_model(
            self, params_phys, data_index, N_vals=N_vals)

    def clear_in_model_likelihood(self) -> None:
        """Deactivate the per-source in-model setup (no-op engines ignore)."""
        self._likelihood_engine.clear_in_model()

    def get_add_ll(self, params, data_index, noise_index, N_vals, phase_maximize=False,
                   leaf_inds=None):
        """Log-likelihood delta of ADDING a source to the model.

        ``ll(r - h) - ll(r) = <r|h> - 0.5 <h|h>`` where ``r`` is the current
        cell residual (which does not contain ``h``). This is a delta, not a
        singular log-likelihood -- the ``d_d`` term cancels. Computed from
        the :attr:`d_h_out` / :attr:`h_h_out` stashed by :meth:`get_ll`;
        :attr:`phase_angle` is available after the call when
        ``phase_maximize=True``. Sources rejected by the engine's bounds
        check (:attr:`kept_out`) come back as ``-1e300``.
        """
        self.get_ll(params, data_index, noise_index, N_vals, phase_maximize=phase_maximize,
                    leaf_inds=leaf_inds)
        delta = self.d_h_out.real - 0.5 * self.h_h_out.real
        delta[~self.kept_out] = -1e300
        return delta

    def get_removal_ll(self, params, data_index, noise_index, N_vals, leaf_inds=None):
        """Log-likelihood delta of REMOVING a source that is in the residual.

        For a residual ``r`` that still *contains* the subtracted template
        ``h`` (i.e. the source is part of the model), the delta of taking it
        out is ``ll(r + h) - ll(r) = -<r|h> - 0.5 <h|h>``. Computed from one
        normal :meth:`get_ll` call by flipping the sign of ``d_h`` -- the
        arithmetic equivalent of evaluating the template with its reference
        phase flipped (``phi -> -phi``, i.e. ``h -> -h``). Delta only; the
        ``d_d`` term cancels. Bounds-rejected sources come back as
        ``-1e300`` (see :attr:`kept_out`).
        """
        self.get_ll(params, data_index, noise_index, N_vals, leaf_inds=leaf_inds)
        delta = -self.d_h_out.real - 0.5 * self.h_h_out.real
        delta[~self.kept_out] = -1e300
        return delta

    def get_replace_ll(self, params_old, params_new, data_index, noise_index,
                       N_vals, phase_maximize=False, leaf_inds=None):
        """Log-likelihood deltas for REPLACING an in-model source (search RJ).

        Self-contained expose -> score -> restore (pure function on the
        residual buffer):

        1. **Expose** the old source: its template is added back into the
           cell residual through :meth:`remove_sources_from_band_buffer`
           (the fixed addremove convention -- removing a source from the
           model ADDS its template to the residual, dev 1928032), so the
           scored residual is ``r' = r + h_old``: the residual as if old
           had never been fit.
        2. **Score** BOTH parameter sets against ``r'`` as add-deltas
           ``<r'|h> - 0.5<h|h>`` in ONE batched :meth:`get_ll` call
           (rows ``[old; new]``), optionally phase-maximized (the engine's
           two-quadrature maximisation; search convention).
        3. **Restore** the touched slots' residual rows from a pre-expose
           snapshot -- bit-exact by construction (a refill with the
           opposite factor would instead round ``(r + h) - h``).

        Row ``i`` of ``params_old`` / ``params_new`` must target the same
        buffer slot ``data_index[i]`` (the old source's cell). Engine
        bounds-rejected rows come back ``-1e300`` (see the ``replace_kept_*``
        attributes).

        Returns:
            ``(delta_old, delta_new, phase_angle_new, delta_old_actual)``:
            add-deltas of the old and new rows vs ``r'``;
            ``phase_angle_new`` is the per-row maximizing rotation of the
            NEW half (``None`` unless ``phase_maximize``);
            ``delta_old_actual`` is the old row's add-delta at its ACTUAL
            phase (equals ``delta_old`` without phase maximisation) -- the
            exact bookkeeping value for an accepted swap's residual-ll
            change, since a rejected/accepted old source is never
            re-phased. Also stashed: :attr:`replace_h_h_old` /
            :attr:`replace_h_h_new` (for SNR clamps) and
            :attr:`replace_kept_old` / :attr:`replace_kept_new`.
        """
        xp = self.xp
        n = params_old.shape[0]

        # (1) snapshot the touched residual rows, then expose the old source.
        slot_rows = xp.unique(xp.asarray(data_index).astype(xp.int64))
        snapshot = self.band_buffer[slot_rows].copy()
        self.remove_sources_from_band_buffer(
            params_old, data_index, N_vals, leaf_inds=leaf_inds
        )
        try:
            # (2) one batched scoring call over [old; new].
            params_cat = xp.concatenate([params_old, params_new], axis=0)
            di_cat = xp.concatenate([data_index, data_index], axis=0)
            ni_cat = xp.concatenate([noise_index, noise_index], axis=0)
            nv_cat = xp.concatenate([N_vals, N_vals], axis=0)
            li_cat = (
                None if leaf_inds is None
                else xp.concatenate([leaf_inds, leaf_inds], axis=0)
            )
            self.get_ll(
                params_cat, di_cat, ni_cat, nv_cat,
                phase_maximize=phase_maximize, leaf_inds=li_cat,
            )
            d_h = self.d_h_out.real.copy()
            h_h = self.h_h_out.real.copy()
            kept = self.kept_out.copy()
            delta = d_h - 0.5 * h_h
            delta[~kept] = -1e300
            phase_angle_new = None
            if phase_maximize and self.phase_angle is not None:
                phase_angle_new = self.phase_angle[n:].copy()
            # Old rows at their ACTUAL phase: the two-quadrature engines
            # stash the un-maximized <r'|h> as ``non_marg_d_h``. Guarded --
            # on a multi-shard route it is not assembled, so fall back to
            # the maximized value (the propose-level ll-drift rebuild then
            # corrects the tracked sum).
            delta_old_actual = delta[:n].copy()
            if phase_maximize:
                _nm = getattr(self._likelihood_engine, "non_marg_d_h", None)
                if _nm is not None and getattr(_nm, "shape", (0,))[0] == 2 * n:
                    delta_old_actual = xp.asarray(_nm).real[:n] - 0.5 * h_h[:n]
                    delta_old_actual[~kept[:n]] = -1e300
        finally:
            # (3) bit-exact restore of the pre-expose residual.
            self.band_buffer[slot_rows] = snapshot

        self.replace_h_h_old = h_h[:n]
        self.replace_h_h_new = h_h[n:]
        self.replace_kept_old = kept[:n]
        self.replace_kept_new = kept[n:]
        return delta[:n], delta[n:], phase_angle_new, delta_old_actual

    def get_ll_grad(self, params, data_index, noise_index, N_vals,
                     *, param_eps=None, chunk=None, leaf_inds=None):
        """Per-source gradient of ``L = <d|h> - 0.5 <h|h>`` w.r.t. params.

        Dispatches to ``self._likelihood_engine.get_ll_grad`` -- only
        the chunked-het backend implements this; the legacy FD path
        raises NotImplementedError. Returns ``(num_proposals, nparams)``
        on the engine's xp module.

        Used by the in-model NUTS / gradient move (the chunked-het
        replacement for the legacy info-matrix Cholesky proposal). The
        buffer must hold the source-of-interest's *clean* residual --
        i.e. ``remove_sources_from_band_buffer`` has been called for
        that source already -- before invoking this.

        The compute backend (C++ central-FD or JAX autograd) is fixed
        on the ``GBWDMComputations`` instance passed in at buffer
        construction time via ``gb_wdm_comp``. Per the sprint-wide
        rule there is no runtime ``backend=`` kwarg; build a JAX-
        backed ``gb_wdm_comp`` if you need the autograd path.
        """
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        return self._likelihood_engine.get_ll_grad(
            self,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            param_eps=param_eps,
            chunk=chunk,
            waveform_kwargs=self.waveform_kwargs,
        )

    def hessian(self, params, data_index, noise_index, N_vals,
                 *, chunk=None,
                 psd_fix=False, psd_floor_rel=1e-30, leaf_inds=None):
        """Per-source Hessian of ``L = <d|h> - 0.5 <h|h>``.

        Dispatches to ``self._likelihood_engine.hessian``. Returns
        ``(num_proposals, nparams, nparams)``. With ``psd_fix=True``,
        returns ``M = |-H|`` (eigendecompose-then-abs, with a relative
        floor) -- ready to feed to ``NUTSSampler(metric=M)`` as a
        per-leaf mass matrix.

        Same buffer-state precondition as :meth:`get_ll_grad`: the
        active source must have been removed from the band buffer
        before calling.

        Currently only the JAX-backed chunked-het generator
        implements ``hessian_wdm``; the C++ chunked-het backend
        raises until the native Hessian kernel lands. Per the
        sprint-wide rule the backend is fixed on the underlying
        ``gb_wdm_comp`` instance -- no runtime ``backend=`` kwarg.
        """
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        return self._likelihood_engine.hessian(
            self,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            chunk=chunk,
            psd_fix=psd_fix,
            psd_floor_rel=psd_floor_rel,
            waveform_kwargs=self.waveform_kwargs,
        )

    def reset_residual_buffers(self, inds_fill=None):
        if inds_fill is None:
            inds_fill = cp.arange(self.num_bands_now)
        self.band_buffer[inds_fill] = 0.0

    def reset_psd_buffers(self, inds_fill=None):
        if inds_fill is None:
            inds_fill = cp.arange(self.num_bands_now)
        self.psd_buffer[inds_fill] = 0.0

    def reset_template_buffers(self, inds_fill=None):
        """Zero the template twin for the given slots.

        The twin is filled ADDITIVELY (engine ``fill_template`` with
        ``factor=+1``), so only a fresh allocation starts it at zero. A
        cached-buffer REBIND (propose-scoped cache 2bd76df / cross-proposal
        persist c94d53c) that skips this reset stacks the new cells'
        templates on top of the previous unit's post-swap contents --
        the root cause of the after-tempering incremental-ll drift
        (sign-consistent ~ -<h|h>/2 per affected cold walker) AND of
        biased tempering swap acceptance (paccept scored against the
        contaminated lls). ``self.xp``, not module-level ``cp`` (the
        CPU-run-with-cupy-importable trap).
        """
        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)
        self.template_buffer[inds_fill] = 0.0

    def fill_buffer_residual_and_psd_from_acs(
        self, acs: AnalysisContainerArray, inds_fill: Optional[cp.ndarray] = None
    ) -> None:
        # CHUNKED (2026-08-14): each tuple-fancy gather below materialises a
        # rows x slab temporary -- for the XYZ invC that is ~763 KB/slot, so
        # a 16,384-row gather is 12.5 GB in ONE allocation (job-183 OOM on a
        # device already holding the persistent buffers). Bound the
        # transient at GB_FILL_CHUNK slots per pass.
        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)
        _chunk = max(1, int(os.environ.get("GB_FILL_CHUNK", "2048")))
        n_fill = int(inds_fill.shape[0])
        if n_fill > _chunk:
            for c0 in range(0, n_fill, _chunk):
                self.fill_buffer_residual_and_psd_from_acs(
                    acs, inds_fill=inds_fill[c0:min(c0 + _chunk, n_fill)])
            return
        # The outer ``acs`` is accessed via tuple-fancy indexing
        # ``data_shaped[0][inds1, inds2, inds3]`` (3-tuple for AET, 5-tuple
        # for XYZ CSD). BandView routes the tuple-fancy index through the
        # owning shard at the right intra-shard band position; on
        # single-shard ACAs the reshape view is touched directly. No
        # outer-buffer materialisation needed.
        # NOTE self.xp, never the module-level ``cp``: on a CPU-backend run
        # on a machine where cupy imports, cp is cupy while the buffer /
        # sorter arrays are numpy (same trap as get_index / the sorter xp).

        # Speed-diagnosis spans (user directive 2026-08-15): fills measured
        # ~8 ms/slot vs ~us of bytes -> decompose index-map construction vs
        # the routed gathers themselves (BandView tuple-fancy indexing may
        # host-stage per shard, same suspect as the get_ll router).
        _ftm = getattr(self, "_prop_timer", None)
        if _ftm is not None:
            _ftm.count("fill_chunks")
            _ftm.count("fill_slots", int(inds_fill.shape[0]))
        outer_data_view = acs.data_shaped_view()
        outer_psd_view = acs.psd_shaped_view()

        with _tspan(_ftm, "fill_indmap_data"):
            inds_get_data = self._get_fill_buffer_ind_map(acs, inds_fill=inds_fill, is_psd=False)

        # load rest of data into buffer (has current sources removed)
        with _tspan(_ftm, "fill_gather_data"):
            self.reset_residual_buffers(inds_fill=inds_fill)

            # By removing `.flatten()` during indexing, broadcasting gives us the exact shape natively.
            band_buf = self.band_buffer
            data_vals = outer_data_view[inds_get_data]
            if isinstance(band_buf, BandView):
                # Per-shard in-place accumulate on each slot's OWNING device
                # (perf, 2026-08): the generic ``+=`` desugared to gather
                # (all target rows onto gpus[0]) + add + scatter — the slot
                # bytes crossed devices twice per fill chunk.
                band_buf.accumulate(inds_fill, data_vals)
            else:
                band_buf[inds_fill] += data_vals
            del data_vals
        del inds_get_data

        with _tspan(_ftm, "fill_indmap_psd"):
            inds_get_psd = self._get_fill_buffer_ind_map(acs, inds_fill=inds_fill, is_psd=True)
        with _tspan(_ftm, "fill_gather_psd"):
            self.reset_psd_buffers(inds_fill=inds_fill)

            psd_vals = outer_psd_view[inds_get_psd]
        psd_buf = self.psd_buffer
        if self.xp.iscomplexobj(psd_vals) and not self.xp.iscomplexobj(
            psd_buf if not isinstance(psd_buf, BandView) else psd_vals
        ):
            # FD buffers store the REAL inverse covariance (gb_fd kernel
            # convention); the parent XYZ CSD invC may be complex.
            psd_vals = psd_vals.real
        psd_buf[inds_fill] = psd_vals
        del inds_get_psd

    def _get_fill_buffer_ind_map(
        self, acs: AnalysisContainerArray, inds_fill: Optional[cp.ndarray] = None, is_psd: bool = False
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:

        if isinstance(self._basis_settings, WDMSettings):
            # WDM fill index map: pick each band's slab out of the parent ACA.
            # Default (full-active) copies the entire (channel, Nf_active,
            # Nt_active) slab. Task-b narrow slabs copy only each slot's own
            # ``band_slab_Nf``-layer window: the parent-local layer offset is
            # ``slab_min_f[slot] - parent.ind_min_f`` (the parent stores active
            # layer ``ind_min_f`` at local index 0), so the layer axis becomes
            # PER-SLOT rather than a shared 0..Nf_active-1 ramp. Data axis
            # position is unique_band_combos[:, 1] (the parent walker index).
            if inds_fill is None:
                inds_fill = self.xp.arange(self.num_bands_now)
            xp = self.xp  # NOT module-level cp (see fill_buffer note)

            Nt_active = self._basis_settings.Nt_active
            slab_Nf = self.band_slab_Nf
            if slab_Nf is None:
                # Full active band: shared layer ramp 0..Nf_active-1.
                Nf_use = self._basis_settings.Nf_active
                layer_lo = xp.zeros(len(inds_fill), dtype=xp.int32)
            else:
                # Narrow: per-slot parent-local layer offset.
                Nf_use = int(slab_Nf)
                layer_lo = (
                    self.slab_min_f[inds_fill] - int(self._basis_settings.ind_min_f)
                ).astype(xp.int32)

            if is_psd and self.tdi_channel_setup == "XYZ":
                # target shape: (len(inds_fill), nchannels, nchannels, Nf_use, Nt_active)
                # The parent WDM ACA's psd_shaped[0] has shape
                # ``(num_walkers, nchan, nchan, Nf_active, Nt_active)`` — one
                # entry per walker with channels as inner axes. Unlike the FD
                # path (which flattens walker*channel into axis 0), here we
                # index axis 0 with the raw walker index and need a full
                # 5-tuple to cover all five axes.
                inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None, None]
                inds2 = xp.arange(self.nchannels)[None, :, None, None, None]
                inds3 = xp.arange(self.nchannels)[None, None, :, None, None]
                inds4 = (layer_lo[:, None, None, None, None]
                         + xp.arange(Nf_use)[None, None, None, :, None])
                inds5 = xp.arange(Nt_active)[None, None, None, None, :]
                return inds1, inds2, inds3, inds4, inds5

            # target shape: (len(inds_fill), nchannels, Nf_use, Nt_active)
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None]
            inds2 = xp.arange(self.nchannels)[None, :, None, None]
            inds3 = (layer_lo[:, None, None, None]
                     + xp.arange(Nf_use)[None, None, :, None])
            inds4 = xp.arange(Nt_active)[None, None, None, :]
            return inds1, inds2, inds3, inds4
        if not isinstance(self._basis_settings, FDSettings):
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

        xp = self.xp  # NOT module-level cp (see fill_buffer note)
        if inds_fill is None:
            inds_fill = xp.arange(self.num_bands_now)

        # ONE evaluation of the parent property (it is a per-AC Python
        # dispatch loop on the ACA); the old code evaluated it three times
        # per map call — twice inside the equality assert alone.
        start_freq_ind_all = acs.start_freq_ind
        start_freq_ind = start_freq_ind_all[0]

        if _index_asserts():
            # Index-bound asserts behind the GB_INDEX_ASSERTS gate (default
            # off in production; the chunked_het kernels use the same knob).
            assert np.all(start_freq_ind == start_freq_ind_all)

            assert np.all((self.buffer_start_index[inds_fill] - start_freq_ind) >= 0), (
                "Buffer start indices fall below the parent data start index."
            )

            assert np.all(
                (self.buffer_start_index[inds_fill] - start_freq_ind + self._fd_store_length)
                <= acs.data_length
            ), f"Buffer indexing exceeds available data length in AnalysisContainerArray. Start indices: {self.buffer_start_index[inds_fill]}, start_freq_ind: {start_freq_ind}, data_length: {self._fd_store_length}, acs data_length: {acs.data_length}"

        start_inds = self.buffer_start_index[inds_fill] - start_freq_ind

        if is_psd and self.tdi_channel_setup == "XYZ":
            # Target output shape:
            # (len(inds_fill), nchannels, nchannels, window_bins).
            # The parent FD ACA's psd_shaped has shape
            # (num_walkers, nchan, nchan, n_bins) -- axis 0 is the raw
            # walker index (mirrors the WDM XYZ 5-tuple below).
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None]
            inds2 = xp.arange(self.nchannels)[None, :, None, None]
            inds3 = xp.arange(self.nchannels)[None, None, :, None]
            inds4 = start_inds[:, None, None, None] + xp.arange(
                self.band_buffer.shape[-1]
            )[None, None, None, :]
            return inds1, inds2, inds3, inds4

        else:
            # Target output shape: (len(inds_fill), self.nchannels, self.band_buffer.shape[-1])
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None]
            inds2 = xp.arange(self.nchannels)[None, :, None]
            inds3 = start_inds[:, None, None] + xp.arange(self.band_buffer.shape[-1])[None, None, :]

        return inds1, inds2, inds3

    def remove_sources_from_template_buffer(self, *args, **kwargs) -> None:
        self._adjust_via_engine(-1, self._acs_template_buffer, *args, **kwargs)

    def add_sources_to_template_buffer(self, *args, **kwargs) -> None:
        self._adjust_via_engine(+1, self._acs_template_buffer, *args, **kwargs)

    def swap_template_slots(self, slots_a, slots_b) -> None:
        """Exchange the template-buffer contents of slot sets ``a`` and ``b``.

        Used by the tempering stage to swap a temperature pair's per-cell
        templates -- and, called again with the rejected subset, to revert
        the swaps that failed the acceptance draw.
        """
        buf = self.template_buffer
        if isinstance(buf, BandView):
            # Multi-GPU: same-shard pairs (the common case under the
            # band-grouped shard assignment) swap in place on their device
            # with no host hop (parallel-resources plan P1).
            buf.swap_rows(slots_a, slots_b)
            return
        tmp = buf[slots_a].copy()
        buf[slots_a] = buf[slots_b]
        buf[slots_b] = tmp[:]

    def _adjust_via_engine(
        self, factor, target_aca, params, params_index, N_vals, *args,
        leaf_inds=None, **kwargs
    ) -> None:
        """Domain-agnostic dispatch into ``self._likelihood_engine.fill_template``.

        ``factor`` is +1 (write source into the template) or -1 (subtract it).
        ``target_aca`` selects which AnalysisContainerArray to write into
        (the buffer itself for residuals, or the template twin). Both share
        the same per-band geometry, so the engine doesn't need to know which
        one it's filling.
        """
        assert isinstance(factor, int) and (factor == -1 or factor == +1)
        params_phys = self._to_phys(params, leaf_inds=leaf_inds)
        # Task-b: forward this buffer's narrow per-band slab metadata so the
        # template write matches the narrow slab layout. Passed ONLY when this
        # is a narrow WDM buffer (band_slab_Nf set) so the FD engine's
        # fill_template -- which takes no such kwargs -- is untouched.
        _slab_kwargs = {}
        if getattr(self, "band_slab_Nf", None) is not None:
            _slab_kwargs = dict(
                band_slab_Nf=self.band_slab_Nf, slab_min_f=self.slab_min_f
            )
        self._likelihood_engine.fill_template(
            target_aca,
            params_phys,
            params_index,
            N_vals,
            factor=factor,
            waveform_kwargs=self.waveform_kwargs,
            **_slab_kwargs,
        )

    def adjust_sources_in_band_buffer(
        self, factor, input_array, params, params_index, N_vals, *args, **kwargs
    ) -> None:
        """Backwards-compatible shim around :meth:`_adjust_via_engine`.

        Routes ``input_array`` (a flat buffer pointer the legacy code passed
        through) back to whichever ACA owns it. New code should call
        :meth:`_adjust_via_engine` directly.
        """
        if input_array is self.band_buffer_tmp:
            target_aca = self
        elif self.use_template_arr and input_array is self.template_buffer_tmp:
            target_aca = self._acs_template_buffer
        else:
            raise ValueError(
                "adjust_sources_in_band_buffer received an input_array that "
                "is neither the band-residual nor the template buffer."
            )
        self._adjust_via_engine(factor, target_aca, params, params_index, N_vals, *args, **kwargs)

    def remove_sources_from_band_buffer(self, *args, **kwargs) -> None:
        # NOTE: sign is +1 because band_buffer holds the residual
        # (= data - sum(templates)). Removing a source from the model means
        # ADDING it back to the residual, hence factor=+1 here.
        self._adjust_via_engine(+1, self, *args, **kwargs)

    def add_sources_to_band_buffer(self, *args, **kwargs) -> None:
        # See remove_sources_from_band_buffer note; sign is flipped for the
        # residual-tracking band_buffer.
        self._adjust_via_engine(-1, self, *args, **kwargs)

    def get_special_band_index(
        self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray
    ) -> np.ndarray:
        return pack_special_index(temp_inds, walker_inds, band_inds, self.nwalkers)

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        return unpack_special_index(special_band_inds, self.nwalkers)


# Back-compat alias: the pre-merge class name.
Buffer = SubBandBuffer


class BandSorter(LISAToolsParallelModule):
    """GPU helper that sorts/ungroups GB samples by frequency band.

    Flattens the eryn GB branch ``(ntemps, nwalkers, nleaves, ndim)`` into
    per-source arrays (``coords`` / ``inds`` / ``temp_inds`` /
    ``walker_inds`` / ``leaf_inds`` / ``band_inds``), assigns each source to
    its frequency band (``searchsorted`` on the *source* frequency -- not the
    domain settings), pre-draws RJ proposals for the ``inds=False`` slots
    when ``rj_prop`` is given, and provides subset / packing machinery for
    the per-band proposal loop and band-temperature swaps.
    """

    @property
    def xp(self) -> Union[ModuleType, numpy, cupy]:
        return self.backend.xp

    @classmethod
    def supported_backends(cls):
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __init__(
        self,
        gb_branch: Branch,
        band_edges: Optional[np.ndarray] = None,
        band_N_vals: Optional[np.ndarray] = None,
        force_backend: Optional[str] = None,
        transform_fn: Optional[TransformContainer] = None,
        copy: bool = True,
        inds_subset: Optional[np.ndarray] = None,
        inds_main_band_sorter: Optional[np.ndarray] = None,
        gb=None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        waveform_kwargs={},
        main_band_sorter=None,
        max_data_store_size: int = 6000,
        rj_prop=None,
        keep_all_inds=True,
        wdm_band_slab_layers: Optional[int] = None,
        wdm_slab_guard_layers: int = 1,
        opt_snr_rej_samp_limit: float = 5.0,
        snr_rej_detected: bool = False,
    ):

        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        self.force_backend = force_backend
        # RJ SNR rejection-sampling clamp (user policy, default 5.0):
        # forwarded into every SubBandBuffer this sorter builds; the copy
        # constructor path below overwrites it with the source sorter's
        # value (attribute copy loop), keeping one source of truth.
        self.opt_snr_rej_samp_limit = float(opt_snr_rej_samp_limit)
        self.snr_rej_detected = bool(snr_rej_detected)

        dc = deepcopy if copy else return_x
        if hasattr(gb_branch, "num_sources"):
            _band_sorter = gb_branch
            self.force_backend = _band_sorter.force_backend
            for key, value in _band_sorter.__dict__.items():
                if key[:2] != "__":
                    if key in [
                        "main_band_sorter",
                        "inds_main_band_sorter",
                        "gb",
                        "gb_wdm_comp",
                        "gb_fd_comp",
                        "rj_prop",
                    ]:
                        continue

                    elif (
                        isinstance(value, self.xp.ndarray)
                        and value.shape[0] == _band_sorter.num_sources
                    ):
                        if inds_subset is None:
                            inds_subset = self.xp.arange(_band_sorter.num_sources)
                        else:
                            assert (
                                isinstance(inds_subset, self.xp.ndarray)
                                and inds_subset.dtype == int
                            )
                            assert inds_subset.max() < (_band_sorter.num_sources)
                        set_value = dc(value[inds_subset])

                    else:
                        set_value = dc(value)

                    setattr(self, key, set_value)

            self.rj_prop = _band_sorter.rj_prop
            self.gb = _band_sorter.gb
            # Forward the computation objects explicitly (skipped in the
            # copy loop so we don't deepcopy GPU-resident objects).
            self.gb_wdm_comp = getattr(_band_sorter, "gb_wdm_comp", None)
            self.gb_fd_comp = getattr(_band_sorter, "gb_fd_comp", None)
            # need to make sure is not mixed up in loop
            self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)
            return

        assert band_edges is not None and band_N_vals is not None
        self.force_backend = force_backend
        self.gb = gb
        # Domain computation objects, forwarded to the buffer in
        # :meth:`get_buffer` so the engine selection matches the parent
        # ACA's basis (WDMSettings -> gb_wdm_comp, FDSettings -> gb_fd_comp).
        self.gb_wdm_comp = gb_wdm_comp
        self.gb_fd_comp = gb_fd_comp
        # Task-b narrow per-band WDM slab extent (None = full active band).
        # Forwarded to each SubBandBuffer built in :meth:`get_buffer`; a plain
        # int/None so the copy-constructor's generic loop carries it through.
        self.wdm_band_slab_layers = wdm_band_slab_layers
        self.wdm_slab_guard_layers = wdm_slab_guard_layers
        self.waveform_kwargs = waveform_kwargs
        self.gb_branch_orig = gb_branch
        self.num_bands = len(band_edges) - 1
        self.band_edges = self.xp.asarray(band_edges)
        self.band_N_vals = self.xp.asarray(band_N_vals)
        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = gb_branch.shape
        self.orig_inds = self.xp.asarray(gb_branch.inds)
        self.keep_all_inds = keep_all_inds
        self.rj_prop = rj_prop

        if rj_prop is not None:
            if keep_all_inds:
                self.coords = self.xp.asarray(gb_branch.coords.reshape(-1, self.ndim))
                self.inds = self.orig_inds.flatten()
            else:
                self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
                self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)

            if self.xp.any(~self.inds):
                # self.xp (run backend), NOT the module-level cp: on a
                # CPU-resolved run on a cupy-installed machine cp is cupy
                # while self.coords is numpy -- mixing them crashes the
                # assignment below (module-cp-vs-force_backend trap).
                new_sources = self.xp.full_like(self.coords[~self.inds], np.nan)
                fix = self.xp.full(new_sources.shape[0], True)
                while self.xp.any(fix):
                    new_sources[fix] = self.xp.asarray(
                        rj_prop.rvs(size=fix.sum().item())
                    )
                    fix = self.xp.any(self.xp.isnan(new_sources), axis=-1)

                self.coords[~self.inds] = new_sources

            proposal_logpdf = self.xp.zeros(self.coords.shape[0])

            batch_here = int(1e6)
            # Batch bounds END at N (eind is exclusive): the old N-1 endpoint
            # silently left the LAST row's proposal logpdf at 0.0, and the
            # [-1] indexing crashed outright on an empty coords array
            # (2026-08-13). np.arange handles N=0 with no special case.
            inds_splitting = np.arange(0, self.coords.shape[0], batch_here)
            if inds_splitting.size == 0 or inds_splitting[-1] != self.coords.shape[0]:
                inds_splitting = np.concatenate(
                    [inds_splitting, np.array([self.coords.shape[0]])]
                )

            for stind, eind in zip(inds_splitting[:-1], inds_splitting[1:]):
                proposal_logpdf[stind:eind] = self.xp.asarray(
                    rj_prop.logpdf(self.coords[stind:eind])
                )
            if self.backend.uses_cupy:
                self.xp.get_default_memory_pool().free_all_blocks()

            if keep_all_inds:
                self.factors = (self.xp.asarray(proposal_logpdf) * -1) * (~self.orig_inds).flatten() + (
                    self.xp.asarray(proposal_logpdf) * +1
                ) * (self.orig_inds).flatten()
                tmp_inds_shaped = self.xp.full_like(self.orig_inds, True)
            else:
                assert self.xp.all(self.inds)
                self.factors = self.xp.asarray(proposal_logpdf) * +1
                tmp_inds_shaped = self.orig_inds.copy()

        else:
            self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
            self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)
            self.factors = self.xp.ones_like(self.inds)
            tmp_inds_shaped = self.orig_inds.copy()

        self.has_run_rj = self.xp.zeros_like(self.inds)
        self.num_sources = self.coords.shape[0]
        self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)

        self.max_data_store_size = max_data_store_size
        self.transform_fn = transform_fn

        self.temp_inds = self.xp.repeat(
            self.xp.arange(self.ntemps), self.nwalkers * self.nleaves_max
        ).reshape(self.ntemps, self.nwalkers, self.nleaves_max)[tmp_inds_shaped]
        self.walker_inds = self.xp.tile(
            self.xp.arange(self.nwalkers), (self.ntemps, self.nleaves_max, 1)
        ).transpose((0, 2, 1))[tmp_inds_shaped]
        self.leaf_inds = self.xp.tile(
            self.xp.arange(self.nleaves_max), ((self.ntemps, self.nwalkers, 1))
        )[tmp_inds_shaped]

        # Per-source frequency: the sampled f0 column when the container has
        # one; the per-leaf f0 fill (Eryn per-leaf fill_dict) otherwise --
        # needs ``leaf_inds``, so computed after the label arrays above.
        self.freqs = self._source_freqs_hz()
        # ⚠ FROZEN FOR THE WHOLE PROPOSE. This is the initial buffer fill:
        # every row (DEAD candidates included -- a birth's band is derived
        # from its own drawn f0 right here, which is why a birth is inside
        # its band by construction) gets the band assignment it keeps until
        # the next propose rebuilds this sorter. A source that drifts
        # across a band edge mid-propose KEEPS this label: its buffer cell,
        # residual add/remove bookkeeping and fill index map are all keyed
        # to it, so re-homing it mid-propose would add and subtract its
        # contribution in different cells and silently corrupt the parent
        # residual. Its assignment changes automatically the next time
        # around, when this line re-runs against its current f0.
        self.band_inds = self.xp.searchsorted(band_edges, self.freqs, side="right") - 1
        self.special_band_inds = self.get_special_band_index(
            self.temp_inds, self.walker_inds, self.band_inds
        )

        self.orig_temp_inds = self.temp_inds.copy()
        self.orig_walker_inds = self.walker_inds.copy()
        self.orig_leaf_inds = self.leaf_inds.copy()
        self.orig_special_band_inds = self.special_band_inds.copy()
        self.orig_band_inds = self.band_inds.copy()

    def _source_freqs_hz(self) -> np.ndarray:
        """Per-source frequency in Hz.

        Sampling-basis f0 is in mHz; when f0 is not a sampled column it must
        be a per-leaf transform fill and each source's value is looked up by
        its leaf index (fixed-source branches, e.g. VGBs).
        """
        tf = self.transform_fn
        input_basis = list(getattr(tf, "input_basis", []) or [])
        if "f0" in input_basis:
            return self.coords[:, input_basis.index("f0")] / 1e3
        if tf is None or getattr(tf, "n_leaf_fills", None) is None:
            # legacy layout without a container: f0 at sampling column 1
            return self.coords[:, 1] / 1e3
        fill_keys = list(tf.original_fill_dict[0].keys())
        if "f0" not in fill_keys:
            raise ValueError(
                "BandSorter: 'f0' is neither a sampled column nor a per-leaf "
                "fill key of the transform container."
            )
        f0_fill_mhz = self.xp.asarray(tf.fill_dict["fill_values"])[
            :, fill_keys.index("f0")
        ]
        return f0_fill_mhz[self.leaf_inds] / 1e3

    def coords_freqs_hz(self, coords):
        """f0 in Hz for an arbitrary block of SAMPLING-basis ``coords``.

        The row-wise twin of :meth:`_source_freqs_hz`, for proposed (not yet
        accepted) parameters -- the leaf-cap cell of an RJ birth is set by
        the DRAWN frequency, not by the dead slot's stale coords. Returns
        ``None`` when f0 is not a sampled column (per-leaf fill branches such
        as VGB): those sources cannot change frequency, so the caller keeps
        the row's stored :attr:`freqs`.
        """
        tf = self.transform_fn
        input_basis = list(getattr(tf, "input_basis", []) or [])
        if "f0" in input_basis:
            return coords[:, input_basis.index("f0")] / 1e3
        if tf is None or getattr(tf, "n_leaf_fills", None) is None:
            return coords[:, 1] / 1e3
        return None

    def set_main_band_sorter_info(self, main_band_sorter, inds_main_band_sorter):
        if main_band_sorter is None:
            self.inds_main_band_sorter = self.xp.arange(self.num_sources)
        else:
            self.inds_main_band_sorter = inds_main_band_sorter

        self.main_band_sorter = main_band_sorter

    @property
    def coords_in(self) -> np.ndarray:
        # leaf_inds is ignored by scalar-fill containers and required by
        # per-leaf-fill ones (Eryn per-leaf fill_dict).
        return self.transform_fn.both_transforms(
            self.coords, xp=self.xp, leaf_inds=self.leaf_inds
        )

    def get_special_band_index(
        self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray
    ) -> np.ndarray:
        return pack_special_index(temp_inds, walker_inds, band_inds, self.nwalkers)

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        return unpack_special_index(special_band_inds, self.nwalkers)

    @property
    def special_index_check(self) -> bool:
        # The standing alarm must judge FLUSHED state: with relabels still
        # pending it would be checking the pre-swap table and pass
        # vacuously, which is the one way this invariant could go quiet.
        self._assert_cell_labels_flushed("special_index_check")
        return self.xp.all(
            self.special_band_inds
            == self.get_special_band_index(self.temp_inds, self.walker_inds, self.band_inds)
        )

    # ------------------------------------------------------------------
    # Deferred cell relabels (GB_CELL_LABEL_DEFERRED)
    # ------------------------------------------------------------------
    # None whenever no window is open. Class-level default so the copy
    # constructor's __dict__ loop, unpickled sorters and every duck-typed
    # test double see the attribute without an __init__ change.
    _deferred_labels = None

    def begin_cell_label_window(self, cells) -> bool:
        """Open a deferred-relabel window over the cells in ``cells``.

        Returns True when the window is armed (knob ON), False when the
        immediate path stays in force -- callers pair it with
        :meth:`flush_cell_labels` unconditionally, since the flush is a
        no-op with no window open.

        THE MODEL (this is the crux -- chained swaps must compose). Inside
        a window SOURCES never change cell; only cells change labels. So
        the window carries a permutation of the cell labels, tracked as
        SLOTS:

        * ``uni[j]``  -- the sorted distinct labels the window was opened
          with; slot ``j`` IS "the sources that carried ``uni[j]`` when
          the window opened", and no source ever leaves its slot;
        * ``cur[j]``  -- slot ``j``'s CURRENT label (plus ``cur_t`` /
          ``cur_w``, the caller-supplied temp/walker that go with it);
        * ``pos[j]``  -- which SLOT currently carries label ``uni[j]``.

        Events name cells by their CURRENT label (the callers pack
        ``(temp, walker, band)`` from live block state), so an event
        resolves its cells through ``pos`` and then writes through ``cur``.
        That indirection is what makes a chain compose: cell A swapped to
        B, then "B" swapped to C, moves A's original sources both times,
        because the second event's lookup of B lands back on A's slot.

        ``cells`` must cover every cell any event in the window can name.
        Both production callers have it exactly: the tempering unit's swap
        grid, and the in-model block's row labels. A cell outside it is a
        programming error, counted device-side (no per-event sync) and
        raised by the flush.
        """
        if not _cell_label_deferred():
            self._deferred_labels = None
            return False
        xp = self.xp
        uni = xp.unique(xp.asarray(cells).flatten())
        m = int(uni.shape[0])
        self._deferred_labels = {
            "uni": uni,
            "cur": uni.copy(),
            "cur_t": xp.zeros(m, dtype=self.temp_inds.dtype),
            "cur_w": xp.zeros(m, dtype=self.walker_inds.dtype),
            "pos": xp.arange(m),
            "dirty": xp.zeros(m, dtype=bool),
            # device-side alarms, read once at the flush
            "bad": xp.zeros((), dtype=xp.int64),
            # HOST-side pending count: the consumer guard reads this, so it
            # must never need a sync (shapes are host metadata).
            "n_pending": 0,
        }
        return True

    def _defer_exchange(self, specials_a, temps_a, walkers_a,
                        specials_b, temps_b, walkers_b) -> None:
        """Compose one pairwise exchange into the open window. O(K)."""
        st = self._deferred_labels
        xp = self.xp
        sa = xp.asarray(specials_a).reshape(-1)
        sb = xp.asarray(specials_b).reshape(-1)
        n = int(sa.shape[0])
        if n == 0 or int(sb.shape[0]) == 0:
            return
        assert int(sb.shape[0]) == n, (
            "BandSorter: deferred exchange needs equal-length cell sets")

        def _wide(v):
            a = xp.asarray(v).reshape(-1)
            return a if int(a.shape[0]) == n else xp.broadcast_to(a, (n,))

        ta, tb = _wide(temps_a), _wide(temps_b)
        wa, wb = _wide(walkers_a), _wide(walkers_b)

        uni = st["uni"]
        m = int(uni.shape[0])
        if m == 0:
            st["bad"] += n
            st["n_pending"] += 1
            return

        # Same band on both sides: the packed key's low digits ARE the band
        # (pack_special_index), so this is the O(K) form of the full-table
        # band assert the immediate path runs under GB_INDEX_ASSERTS -- no
        # table scan, so it can stay armed whenever that knob is.
        if _index_asserts():
            st["bad"] += (sa % _SPECIAL_INDEX_BASE
                          != sb % _SPECIAL_INDEX_BASE).sum()

        j_a = xp.clip(xp.searchsorted(uni, sa), 0, m - 1)
        j_b = xp.clip(xp.searchsorted(uni, sb), 0, m - 1)
        # membership: accumulate on device, surface at the flush (a sync
        # here would reintroduce exactly the per-event stall being removed)
        st["bad"] += (uni[j_a] != sa).sum() + (uni[j_b] != sb).sum()

        # slots currently carrying the two labels
        i_a = st["pos"][j_a]
        i_b = st["pos"][j_b]

        st["cur"][i_a] = sb
        st["cur_t"][i_a] = tb
        st["cur_w"][i_a] = wb
        st["cur"][i_b] = sa
        st["cur_t"][i_b] = ta
        st["cur_w"][i_b] = wa
        st["dirty"][i_a] = True
        st["dirty"][i_b] = True
        # label uni[j_b] now sits in slot i_a, and uni[j_a] in slot i_b
        st["pos"][j_b] = i_a
        st["pos"][j_a] = i_b
        st["n_pending"] += 1

    def _assert_cell_labels_flushed(self, where: str) -> None:
        """Guard for a consumer that must not read a deferred label.

        Host-side only (``n_pending`` is a python int), so it is free and
        can stay armed in production -- silent until something actually
        pends, loud the moment a consumer would read a label the window has
        already moved.
        """
        st = self._deferred_labels
        assert st is None or st["n_pending"] == 0, (
            f"BandSorter: {where} would read cell labels with "
            f"{st['n_pending']} deferred relabel(s) pending -- "
            "flush_cell_labels() before this consumer "
            "(GB_CELL_LABEL_DEFERRED)."
        )

    def flush_cell_labels(self, close: bool = False) -> bool:
        """Apply the window's accumulated permutation in ONE table pass.

        Returns True when a window was open (whether or not anything
        pended), False when there was none -- so an unconditional call at a
        flush point is safe with the knob OFF.

        ``close=False`` (the per-chunk flush point) applies and RE-ANCHORS:
        the slots are re-based onto the labels the table now holds, which
        is exactly the same invariant the window opened with, so the window
        keeps running. ``close=True`` ends it.
        """
        st = self._deferred_labels
        if st is None:
            return False
        xp = self.xp
        # ``bad`` only ever grows inside _defer_exchange, which also bumps
        # the host-side n_pending, so a window with nothing pending needs
        # no device read at all -- an empty chunk costs zero syncs.
        if st["n_pending"]:
            if int(st["bad"]) != 0:
                self._deferred_labels = None
                raise AssertionError(
                    "BandSorter: a deferred cell relabel named a cell "
                    "outside the window's declared universe, or exchanged "
                    "cells across bands -- the label table is inconsistent "
                    "(GB_CELL_LABEL_DEFERRED)."
                )
            dirty = st["dirty"]
            # uni is sorted and a boolean mask preserves order, so src is
            # sorted -- no argsort needed (the immediate primitive pays one)
            src = st["uni"][dirty]
            if int(src.shape[0]) > 0:
                dst_spec = st["cur"][dirty]
                dst_t = st["cur_t"][dirty]
                dst_w = st["cur_w"][dirty]
                keep = xp.isin(self.special_band_inds, src)
                take = xp.searchsorted(
                    src, self.special_band_inds[keep], side="left")
                self.special_band_inds[keep] = dst_spec[take]
                self.temp_inds[keep] = dst_t[take]
                self.walker_inds[keep] = dst_w[take]
        if close:
            self._deferred_labels = None
            return True
        m = int(st["uni"].shape[0])
        st["cur"] = st["uni"].copy()
        st["pos"] = xp.arange(m)
        st["dirty"] = xp.zeros(m, dtype=bool)
        st["n_pending"] = 0
        return True

    def exchange_cell_labels(self, specials_a, temp_a, walkers_a,
                             specials_b, temp_b, walkers_b, bands=None) -> None:
        """Pairwise-swap the (temp, walker) labels of the sources in cell
        sets ``a`` and ``b``.

        ``specials_a[k]`` exchanges with ``specials_b[k]``: every source in
        cell ``a_k`` is relabelled to ``(temp_b, walkers_b[k])`` and vice
        versa (band indices are unchanged -- tempering swaps stay within a
        band; pass ``bands`` to assert that). Both membership maps are
        computed BEFORE any mutation so the two directions cannot see each
        other's relabelled sources.

        With a deferred window open (:meth:`begin_cell_label_window`) this
        composes into the window instead and the table is untouched until
        the flush; the observable end state is identical.
        """
        if self._deferred_labels is not None:
            self._defer_exchange(specials_a, temp_a, walkers_a,
                                 specials_b, temp_b, walkers_b)
            return
        xp = self.xp

        def _map(specials_from):
            order = xp.argsort(specials_from.flatten())
            keep = xp.isin(self.special_band_inds, specials_from)
            take = order[xp.searchsorted(
                specials_from[order], self.special_band_inds[keep], side="left"
            )]
            return keep, take

        keep_a, take_a = _map(specials_a)   # take_* indexes the OTHER set's rows
        keep_b, take_b = _map(specials_b)

        # Gated (2026-08-27 tempering audit): these two device-bool asserts
        # synced the host on EVERY rung-pair step (~40k/iteration). The
        # once-per-unit special_index_check in run_tempering is the
        # standing alarm; arm these per-call with GB_INDEX_ASSERTS=1.
        if bands is not None and _index_asserts():
            assert xp.all(self.band_inds[keep_a] == bands[take_a])
            assert xp.all(self.band_inds[keep_b] == bands[take_b])

        self.special_band_inds[keep_a] = specials_b[take_a]
        self.temp_inds[keep_a] = temp_b
        self.walker_inds[keep_a] = walkers_b[take_a]

        self.special_band_inds[keep_b] = specials_a[take_b]
        self.temp_inds[keep_b] = temp_a
        self.walker_inds[keep_b] = walkers_a[take_b]

    def exchange_cell_labels_batch(self, specials_a, temps_a, walkers_a,
                                   specials_b, temps_b, walkers_b,
                                   bands=None) -> None:
        """Vectorized pairwise exchange for K DISJOINT cell pairs.

        Semantically identical to K sequential
        :meth:`exchange_cell_labels` calls with single-cell sets --
        ``specials_a[k]``'s sources take labels ``(temps_b[k],
        walkers_b[k], specials_b[k])`` and vice versa -- PROVIDED the 2K
        cells are pairwise disjoint (each cell in at most one pair). The
        vertical sweep's cold-rung parity selection and the tempering
        chunk grid both guarantee that.

        WHY (orchestration audit 2026-08-27): the per-pair loop cost 2
        full-table ``isin`` + 2 ``int()`` syncs + 2 assert syncs PER
        ACCEPTED SWAP (~51 of the 70 ms/repeat-step). This does ONE
        membership pass for all pairs. The per-call band assert is gated
        behind ``GB_INDEX_ASSERTS`` -- the block-end
        ``special_index_check`` remains the standing regression alarm.

        With a deferred window open (:meth:`begin_cell_label_window`) this
        composes into the window instead and the table is untouched until
        the flush; the observable end state is identical.
        """
        if self._deferred_labels is not None:
            self._defer_exchange(specials_a, temps_a, walkers_a,
                                 specials_b, temps_b, walkers_b)
            return
        xp = self.xp
        src = xp.concatenate(
            [xp.asarray(specials_a).flatten(),
             xp.asarray(specials_b).flatten()])
        if int(src.shape[0]) == 0:
            return
        dst_spec = xp.concatenate(
            [xp.asarray(specials_b).flatten(),
             xp.asarray(specials_a).flatten()])
        dst_temp = xp.concatenate(
            [xp.asarray(temps_b).flatten(), xp.asarray(temps_a).flatten()])
        dst_walk = xp.concatenate(
            [xp.asarray(walkers_b).flatten(),
             xp.asarray(walkers_a).flatten()])
        order = xp.argsort(src)
        keep = xp.isin(self.special_band_inds, src)
        take = order[xp.searchsorted(
            src[order], self.special_band_inds[keep], side="left")]
        if bands is not None and _index_asserts():
            bb = xp.concatenate(
                [xp.asarray(bands).flatten(), xp.asarray(bands).flatten()])
            assert xp.all(self.band_inds[keep] == bb[take])
        self.special_band_inds[keep] = dst_spec[take]
        self.temp_inds[keep] = dst_temp[take]
        self.walker_inds[keep] = dst_walk[take]

    @property
    def N_vals(self) -> np.ndarray:
        return self.band_N_vals[self.band_inds]

    @property
    def unique_N(self) -> np.ndarray:
        return self.xp.unique(self.N_vals)

    def get_subset(self, *args, **kwargs):
        # A subset COPIES the label arrays (the __dict__ loop above), so a
        # relabel still pending in a deferred window would never reach it.
        self._assert_cell_labels_flushed("get_subset")
        subset_inds = self.get_subset_inds(*args, **kwargs)

        if len(subset_inds) == 0:
            return None

        # source information
        subset = BandSorter(
            self,
            inds_subset=subset_inds,
            main_band_sorter=self.main_band_sorter,
            inds_main_band_sorter=self.inds_main_band_sorter[subset_inds],
        )
        # band information
        return subset

    def get_subset_inds(self, *args, **kwargs):
        subset_bool = self.get_subset_bool(*args, **kwargs)
        return self.xp.arange(len(subset_bool))[subset_bool]

    def get_subset_bool(
        self,
        units: Optional[int] = None,
        remainder: Optional[int] = None,
        temp: Optional[int] = None,
        walker: Optional[int] = None,
        leaf: Optional[int] = None,
        band: Optional[int] = None,
        apply_inds: Optional[bool] = False,
        special_band_inds: Optional[int | np.ndarray] = None,
        extra_bool: Optional[np.ndarray] = None,
        full_bool: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        self._assert_cell_labels_flushed("get_subset_bool")
        inds_keep = self.xp.ones_like(self.band_inds, dtype=bool)

        if full_bool is None:
            if band is not None:
                assert isinstance(band, int)
                inds_keep &= self.band_inds == band
            elif units is not None or remainder is not None:
                assert units is not None and remainder is not None
                inds_keep &= self.band_inds % units == remainder

            if temp is not None:
                assert isinstance(temp, int)
                inds_keep &= self.temp_inds == temp
            if walker is not None:
                assert isinstance(walker, int)
                inds_keep &= self.walker_inds == walker
            if leaf is not None:
                assert isinstance(leaf, int)
                inds_keep &= self.leaf_inds == leaf

            if extra_bool is not None:
                assert isinstance(extra_bool, self.xp.ndarray)
                assert extra_bool.shape == (self.num_sources,)
                inds_keep &= extra_bool

            if apply_inds:
                inds_keep &= self.inds

            if special_band_inds is not None:
                if isinstance(special_band_inds, int):
                    inds_keep &= self.special_band_inds == special_band_inds

                elif isinstance(special_band_inds, self.xp.ndarray):
                    inds_keep &= self.xp.isin(self.special_band_inds, special_band_inds)

        else:
            assert full_bool.shape[0] == self.num_sources
            inds_keep = full_bool

        return inds_keep

    @property
    def main_band_sorter(self):
        main_band_sorter = self if self._main_band_sorter is None else self._main_band_sorter
        return main_band_sorter

    @main_band_sorter.setter
    def main_band_sorter(self, main_band_sorter):
        self._main_band_sorter = main_band_sorter

    def get_buffer(
        self, acs, special_indices_unique, inds_fill=None, buffer_obj=None,
        allow_resize: bool = False, timer=None, fill_slots=None, **kwargs
    ) -> SubBandBuffer:
        """Build or rebind a :class:`SubBandBuffer` for these cells.

        ``timer``: optional ``_ProposeTimer`` (buffer-work sub-marks,
        2026-08-14); phases are wrapped in the spans ``bufbuild_alloc``
        (fresh SubBandBuffer construction) / ``buffill_resid_psd``
        (residual+PSD fill from the parent ACs) / ``buffill_inject``
        (source injection into the band buffers) / ``buffill_template``
        (template-twin reset + template injection). They are reported
        nested inside the caller's buffer_build / temper_buffer span totals
        so a production log decomposes alloc vs fill vs injection vs
        template generation. ``None`` (default) is a no-op.

        ``fill_slots``: optional SUBSET of ``inds_fill`` (slot indices) that
        actually receives the residual/PSD copy and the template-twin reset.
        The binding itself (specials map, source maps, band combos) is still
        built over the FULL ``inds_fill`` -- only the per-slot slab traffic
        is restricted. The slots left out keep whatever their slabs held
        before, so a caller may pass this ONLY for cells it will never score
        and never inject into (the tempering stage's sourceless rows; see
        ``GBSpecialBase.run_tempering`` / ``GB_TEMPER_SKIP_EMPTY``).
        ``None`` (default) fills every bound slot -- today's behavior.
        """

        num_band_preload = len(special_indices_unique)

        # ``sources_now_map`` below is a row-index map built FROM the labels,
        # and the buffer freezes it -- a pending deferred relabel would bind
        # the wrong rows for the whole life of the buffer.
        self.main_band_sorter._assert_cell_labels_flushed("get_buffer")

        # Array module from the SORTER's arrays, not the module-level ``cp``:
        # on a CPU run on a machine where cupy imports (cluster), ``cp`` is
        # cupy while the sorter holds numpy -- cp.isin then dies with
        # ``TypeError: Unsupported type <class 'numpy.ndarray'>`` (and
        # cp.arange/cp.asarray would silently UPLOAD host data instead).
        xp = get_array_module(self.main_band_sorter.special_band_inds)

        # CAN USE main_band_sorter TO GET SOURCES IN BANDS OF INTEREST THAT ARE NOT CURRENTLY OF INTEREST THEMSELVES

        sources_now_map = xp.arange(self.main_band_sorter.special_band_inds.shape[0])[
            xp.isin(self.main_band_sorter.special_band_inds, special_indices_unique)
        ]

        # NOTE: self.main_band_sorter.inds needed to only inject real sources
        # inject sources must include sources that have been turned off in these bands
        sources_inject_now_map = xp.arange(self.main_band_sorter.special_band_inds.shape[0])[
            xp.isin(self.main_band_sorter.special_band_inds, special_indices_unique)
            & self.main_band_sorter.inds
        ]

        # separate out inds
        temp_inds_now, walker_inds_now, band_inds_now = self.get_separate_inds_from_special_index(
            special_indices_unique
        )

        all_unique_band_combos = xp.asarray([temp_inds_now, walker_inds_now, band_inds_now]).T
        num_bands_here_total = all_unique_band_combos.shape[0]
        num_bands_now = special_indices_unique.shape[0]

        points_curr_tmp = self.main_band_sorter.coords[sources_now_map].copy()
        curr_special_band_inds = self.main_band_sorter.special_band_inds[sources_now_map].copy()

        # sort these sources by band
        if inds_fill is None:
            inds_fill = xp.arange(num_band_preload)
            assert buffer_obj is None
            with _tspan(timer, "bufbuild_alloc"):
                buffer_obj = SubBandBuffer(
                    self.rj_prop,
                    self.nwalkers,
                    self.gb,
                    self.band_edges,
                    self.band_N_vals,
                    all_unique_band_combos,
                    points_curr_tmp,
                    num_bands_now,
                    acs.nchannels,
                    self.max_data_store_size,
                    special_indices_unique,
                    self.transform_fn,
                    self.waveform_kwargs,
                    # Frequency spacing for the band-index math. FDSettings uses
                    # the FD bin resolution ``.df``; WDMSettings uses
                    # ``.layer_df`` so the ``band_edges / df`` math yields WDM
                    # *layer* indices -- the same quantity the WDM likelihood
                    # engine addresses by (WDMBandLikelihoodEngine uses
                    # ``basis_settings.layer_df``).
                    (acs.settings.df if isinstance(acs.settings, FDSettings)
                     else acs.settings.layer_df),
                    sources_now_map,
                    sources_inject_now_map,
                    self.main_band_sorter.special_band_inds[sources_now_map],
                    basis_settings=acs.settings,
                    gb_wdm_comp=self.gb_wdm_comp,
                    gb_fd_comp=self.gb_fd_comp,
                    force_backend=self.force_backend,
                    wdm_band_slab_layers=self.wdm_band_slab_layers,
                    wdm_slab_guard_layers=self.wdm_slab_guard_layers,
                    opt_snr_rej_samp_limit=getattr(
                        self, "opt_snr_rej_samp_limit", 5.0),
                    snr_rej_detected=getattr(
                        self, "snr_rej_detected", False),
                    **kwargs,
                )

        else:
            assert isinstance(buffer_obj, SubBandBuffer)
            if (
                allow_resize
                and getattr(buffer_obj, "alloc_capacity", None) is not None
                and int(num_bands_now) != int(buffer_obj.num_bands_now)
            ):
                # Resize-rebind (fixed-capacity buffer, user ruling
                # 2026-08-14): rebind this unit's k cells into the front of
                # the capacity allocation instead of dropping + rebuilding.
                # ``allow_resize`` is an EXPLICIT opt-in from
                # _cached_get_buffer — an in-round partial rotation also has
                # len(specials) != num_bands_now and must never trigger a
                # resize. The caller passes inds_fill = arange(k), so the
                # full-rebind branch below (maps refresh) always runs after
                # a resize.
                assert int(num_bands_now) <= int(buffer_obj.alloc_capacity), (
                    f"resize-rebind of {int(num_bands_now)} cells exceeds "
                    f"alloc_capacity={int(buffer_obj.alloc_capacity)}"
                )
                assert int(inds_fill.shape[0]) == int(num_bands_now), (
                    "resize-rebind requires inds_fill spanning the full new "
                    "binding (arange(k))."
                )
                buffer_obj.resize_to(int(num_bands_now))
            assert inds_fill.max() <= buffer_obj.num_bands_now
            # THIS NEEDS TO HAPPEN before updating data
            buffer_obj.update_special_indices(special_indices_unique, inds_fill=inds_fill)
            if int(inds_fill.shape[0]) == int(buffer_obj.num_bands_now):
                # FULL rebind (cross-unit reuse of a cached buffer: same
                # allocation, new parity unit): refresh the subset-derived
                # source maps so the rebind is exactly construction minus
                # allocation. Partial rebinds (in-round slot rotation) must
                # NOT touch these -- their maps cover only the rotated slots.
                buffer_obj.sources_now_map = sources_now_map
                buffer_obj.sources_inject_now_map = sources_inject_now_map
                buffer_obj.params_interest = points_curr_tmp
                buffer_obj.special_band_inds = curr_special_band_inds
                buffer_obj.now_index = buffer_obj.get_index(curr_special_band_inds)

        # Slab-traffic slots: the full binding unless the caller restricted
        # it (empty-cell skip). Everything above this point -- allocation,
        # specials/source maps, band combos -- always covers all of
        # ``inds_fill``, so the buffer stays fully addressable either way.
        slab_fill = inds_fill if fill_slots is None else fill_slots
        if fill_slots is not None and _index_asserts():
            assert bool(xp.all(xp.isin(fill_slots, inds_fill))), (
                "get_buffer(fill_slots=...) must be a subset of inds_fill."
            )

        with _tspan(timer, "buffill_resid_psd"):
            # A fully-skipped binding (every bound cell sourceless) copies
            # nothing at all -- the empty index arrays would otherwise walk
            # the tuple-fancy shard router for zero bytes.
            if int(slab_fill.shape[0]) > 0:
                buffer_obj.fill_buffer_residual_and_psd_from_acs(
                    acs, inds_fill=slab_fill
                )
        buffer_obj.parent_acs = acs
        # includes sources in these sub-bands that are no longer getting proposals
        coords_to_inject = self.main_band_sorter.coords[sources_inject_now_map].copy()
        inj_special_indices_now = self.main_band_sorter.special_band_inds[
            sources_inject_now_map
        ].copy()

        inject_index = buffer_obj.get_index(inj_special_indices_now)
        inject_N_vals = self.band_N_vals[
            self.main_band_sorter.band_inds[sources_inject_now_map]
        ].copy()

        assert len(inject_index) == len(coords_to_inject)

        # leaf identity threads the per-leaf transform fills (ignored by
        # scalar-fill containers)
        inject_leaf_inds = self.main_band_sorter.leaf_inds[
            sources_inject_now_map
        ].copy()
        inj_args = (coords_to_inject, inject_index, inject_N_vals)
        if buffer_obj.use_template_arr:
            with _tspan(timer, "buffill_template"):
                # The twin fill below is additive: without this reset a
                # cached-buffer rebind inherits the previous bind's
                # (post-swap) templates and every ll scored from the twin is
                # contaminated. No-op cost on a fresh allocation (already
                # zero).
                if int(slab_fill.shape[0]) > 0:
                    buffer_obj.reset_template_buffers(inds_fill=slab_fill)
                buffer_obj.add_sources_to_template_buffer(
                    *inj_args, leaf_inds=inject_leaf_inds
                )
        else:
            with _tspan(timer, "buffill_inject"):
                buffer_obj.add_sources_to_band_buffer(
                    *inj_args, leaf_inds=inject_leaf_inds
                )

        return buffer_obj

    # ------------------------------------------------------------------
    # Group-stretch friends
    # ------------------------------------------------------------------

    def build_friend_index(self, nfriends: int) -> bool:
        """Build the sorted cold-chain frequency table + per-source friend windows.

        Friends are cold-chain (``temp == 0``, ``inds == True``) sources close
        in frequency: for every source (any temperature) we store the start of
        an ``nfriends``-wide window into the frequency-sorted cold-chain
        coordinate table, centred on the source's own frequency (clamped at
        the table edges). :meth:`draw_friends` then draws one friend uniformly
        from that window per source.

        Returns ``False`` (and clears the table) when there are too few
        cold-chain sources to form a window.
        """
        table = self.build_friend_table(nfriends)
        return self.index_friends(table, nfriends)

    def build_friend_table(self, nfriends: int):
        """The frequency-sorted cold-chain coordinate table, or ``None``.

        Split out of :meth:`build_friend_index` so the table can be built ONCE
        per larger iteration and shared across the GB moves of a cycle, while
        :meth:`index_friends` still runs per proposal (each proposal's sorter
        holds a different set of sources).
        """
        self._assert_cell_labels_flushed("build_friend_table")
        cold = self.inds & (self.temp_inds == 0)
        if int(cold.sum()) < max(2, int(nfriends)):
            return None
        cold_coords = self.coords[cold]
        order = self.xp.argsort(cold_coords[:, 1])
        return cold_coords[order].copy()

    def index_friends(self, friends_coords_sorted, nfriends: int) -> bool:
        """Window every source into ``friends_coords_sorted`` (per proposal)."""
        self.nfriends = int(nfriends)
        if friends_coords_sorted is None or len(friends_coords_sorted) < max(
            2, self.nfriends
        ):
            self.friend_start_inds = None
            return False

        n_cold = len(friends_coords_sorted)
        self.friends_coords_sorted = friends_coords_sorted
        self.friends_freqs_sorted = friends_coords_sorted[:, 1].copy()

        starts = (
            self.xp.searchsorted(self.friends_freqs_sorted, self.coords[:, 1], side="right")
            - self.nfriends // 2
        )
        self.friend_start_inds = self.xp.clip(starts, 0, n_cold - self.nfriends).astype(
            self.xp.int32
        )
        return True

    def draw_friends(self, source_ids):
        """One random friend (coordinate row) per source in ``source_ids``.

        Requires :meth:`build_friend_index` to have been run this proposal.
        """
        starts = self.friend_start_inds[source_ids]
        deviation = self.xp.random.randint(0, self.nfriends, size=len(starts))
        take = self.xp.clip(starts + deviation, 0, len(self.friends_freqs_sorted) - 1)
        return self.friends_coords_sorted[take]

    # ------------------------------------------------------------------
    # Shared info-matrix Cholesky table (group-stretch friends analogue)
    # ------------------------------------------------------------------

    def build_infomat_index(self, infomat_freqs_sorted, infomat_chol_sorted) -> bool:
        """Map every source onto the NEAREST entry of a shared Cholesky table.

        The direct counterpart of :meth:`build_friend_index`: that method
        windows each source into a frequency-sorted table of cold-chain
        COORDINATES, this one points each source at the single closest entry
        of a frequency-sorted table of cold-chain proposal CHOLESKY factors
        (built by the move every N iterations, since it needs the likelihood
        model the sorter has no handle on).  Sources at every temperature and
        walker index into the same cold-chain table, exactly as friends do.

        Returns ``False`` (and clears the index) when the supplied table is
        empty, so the caller can fall back to computing factors directly.
        """
        if infomat_freqs_sorted is None or len(infomat_freqs_sorted) == 0:
            self.infomat_take_inds = None
            return False

        self.infomat_freqs_sorted = infomat_freqs_sorted
        self.infomat_chol_sorted = infomat_chol_sorted
        n_tab = len(infomat_freqs_sorted)

        # searchsorted gives the right-hand neighbour; pick whichever of the
        # bracketing pair is closer in frequency (clamped at the table edges).
        hi = self.xp.searchsorted(infomat_freqs_sorted, self.coords[:, 1], side="left")
        hi = self.xp.clip(hi, 0, n_tab - 1)
        lo = self.xp.clip(hi - 1, 0, n_tab - 1)
        d_hi = self.xp.abs(infomat_freqs_sorted[hi] - self.coords[:, 1])
        d_lo = self.xp.abs(infomat_freqs_sorted[lo] - self.coords[:, 1])
        self.infomat_take_inds = self.xp.where(d_lo <= d_hi, lo, hi).astype(
            self.xp.int32
        )
        return True

    def draw_infomat(self, source_ids):
        """Nearest-in-frequency Cholesky factor per source in ``source_ids``.

        Requires :meth:`build_infomat_index` to have been run this proposal;
        mirrors :meth:`draw_friends`, but the pick is deterministic (nearest)
        rather than a uniform draw from a window.
        """
        take = self.infomat_take_inds[source_ids]
        return self.infomat_chol_sorted[take]

    def get_band_info(self):

        self._assert_cell_labels_flushed("get_band_info")
        uni_special, uni_special_counts = cp.unique(
            self.special_band_inds[self.inds], return_counts=True
        )
        uni_temp_inds, uni_walker_inds, uni_band_inds = self.get_separate_inds_from_special_index(
            uni_special
        )

        num_bands = len(self.band_edges) - 1
        band_counts = np.zeros((self.ntemps, self.nwalkers, num_bands), dtype=int)
        band_counts[_to_numpy(uni_temp_inds), _to_numpy(uni_walker_inds), _to_numpy(uni_band_inds)] = (
            _to_numpy(uni_special_counts)
        )

        return {"band_counts": band_counts}
