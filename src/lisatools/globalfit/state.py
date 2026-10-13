"""Per-branch sampler-state subclasses used by the global fit."""

import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from eryn.state import Branch as eryn_Branch
from eryn.state import State as eryn_State

logger = logging.getLogger(__name__)


def return_x(x):
    """Identity helper used as a no-op replacement for :func:`copy.deepcopy`."""
    return x


def _scalar_or_none(value):
    """``value`` as an int if it is scalar-like, else ``None``.

    The main backend's flat kwargs merge can hand branch-keyed dicts (e.g.
    ``nleaves_max={branch: n}``) to a sub-backend reset; those are not this
    branch's dimensions and are treated as absent.
    """
    if value is None or isinstance(value, dict):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def branch_nleaves_max(possible_state, name: str) -> int:
    """``nleaves_max`` for branch ``name`` from a coords-like dict OR an eryn ``State``.

    The per-branch sub-state constructors historically assumed a
    ``{branch: coords}`` dict input, but on HDF reload
    (``GFHDFBackend.get_a_sample``) they receive a plain
    :class:`eryn.state.State`, which is not subscriptable. Both carry the
    ``(ntemps, nwalkers, nleaves_max, ndim)`` shape; dispatch on which one
    arrived.
    """
    branches = getattr(possible_state, "branches", None)
    if branches is not None:
        return int(branches[name].shape[-2])
    return int(possible_state[name].shape[-2])


def make_cap_edges(band_edges, divisor: int, stagger: bool = False):
    """Subdivide every sub-band into ``divisor`` equal-width CAP CELLS.

    The leaf-cap grid (user design 2026-08-15). Sub-band widths are set by
    what the likelihood engine can run concurrently; the scale at which
    sources actually get confused is the POSTERIOR width, which is much
    narrower. So the caps move off the band grid onto a finer one, while
    scheduling / units / buffers / tempering / band shutoff all stay on the
    band grid.

    Default (``stagger=False``): every cap cell is CONTAINED in exactly one
    sub-band (band ``b`` owns cells ``b*K ... b*K + K - 1``). That
    containment is what keeps the bookkeeping and the HDF5 storage simple
    and makes the construction work unchanged under both band-edge modes
    (``uniform`` and the free-floating ``get_n`` grid).

    ``stagger=True`` (user design 2026-08-20, the v5 grid): every interior
    cap edge is shifted by HALF a cell width, so no cap edge coincides with
    a band edge -- the two grids share no equivalent boundaries and no
    source can sit on a band seam and a cap seam simultaneously. Band ``b``
    still OWNS cells ``b*K ... b*K + K - 1`` (index arithmetic, reshapes
    and array sizes are all unchanged), but the cell at index ``b*K``
    (``b > 0``) physically STRADDLES the band-``(b-1)``/``b`` boundary:
    it spans the top half-cell of band ``b-1`` plus the bottom half-cell
    of band ``b``. The first cell of the grid is a half-width cell and the
    last is 1.5 cells wide (its would-be top edge is dropped so the edge
    and cell counts match the nested grid exactly). Cell membership of a
    frequency in band ``b`` is ``b*K + floor((f - lo_b)/step_b + 1/2)``,
    clipped to the global cell range -- see the move's ``_cap_cell_index``.

    ``divisor == 1`` UNSTAGGERED returns a copy of ``band_edges`` itself
    (the cap grid IS the band grid), so every downstream cap computation
    reduces bit-identically to the pre-2026-08-15 per-band behaviour.

    ``divisor == 1`` WITH ``stagger`` is the MIDPOINT-TO-MIDPOINT grid
    (user design 2026-08-29: "the cap cell should go from the midpoint of
    1 sub-band to the midpoint of the next; there should be approximately
    the same number of cap cells and sub-bands, with some slight
    adjustment on the edges"). Edges are ``[be[0], mid_0, ..., mid_-2,
    be[-1]]`` -- one cell per sub-band, each interior cell straddling
    exactly ONE band seam, a half-width first cell and a 1.5-width last
    one. This is the grid that makes the two sides of a seam compete for
    a single cap WITHOUT halving the cell width (``divisor=2`` cells are
    half a sub-band wide, which doubles the cell count, halves per-cell
    occupancy and thereby DELAYS at-cap birth-row exclusion -- measured
    at +39% F-stat candidate rows on the 2026-08-29 v7 restart).

    It also matters for ENFORCEMENT: ``_cap_at_cap_mask`` tests a dead
    (birth-candidate) row against BAND saturation at ``K >= 2`` -- "is
    every cap cell of my band full" -- and a band's ownership of cells
    ``b*K ... b*K+K-1`` is index arithmetic, so under a staggered grid the
    cell a birth physically LANDS in may belong to a neighbour's index
    range and be at cap while the birth's own band still has headroom
    (observed: 4 of 24 walkers holding 2 leaves in a cap-1 straddling
    cell, v7 rows 5-6). At ``K == 1`` that function returns the own-cell
    test for every row, so the destination cell is what gates the birth.

    Args:
        band_edges: 1D ascending array of sub-band edges (Hz).
        divisor: Number of cap cells per sub-band (``K >= 1``).
        stagger: Shift interior cap edges by half a cell so no cap edge
            equals a band edge (requires ``K >= 2``; ignored at ``K == 1``).

    Returns:
        ``np.ndarray`` of length ``K * num_bands + 1`` in both modes.
    """
    be = np.asarray(band_edges, dtype=float)
    k = max(1, int(divisor))
    if k == 1 and not stagger:
        return be.copy()
    lo = be[:-1]
    step = (be[1:] - be[:-1]) / k
    if stagger:
        inner = lo[:, None] + (np.arange(k)[None, :] + 0.5) * step[:, None]
        return np.concatenate([be[:1], inner.ravel()[:-1], be[-1:]])
    inner = lo[:, None] + np.arange(k)[None, :] * step[:, None]
    return np.concatenate([inner.ravel(), be[-1:]])


def make_cap_edge_extensions(band_edges, cap_edges, divisor, overlap_frac):
    """Per-EDGE half-extensions for OVERLAPPING cap cells (design 2026-08-23).

    Overlap mode widens every cap cell's enforcement SPAN symmetrically --
    cell ``i`` covers ``[e_i - x_i, e_{i+1} + x_{i+1}]`` -- while the stored
    edge array itself NEVER changes (same count, stride and stagger, so
    resume guards and stored cap arrays keep their shapes). The extension is
    set so adjacent cells share a fraction ``p = overlap_frac`` of the
    cell's own width ``w`` with each neighbour::

        w = s / (1 - p)          # s = cap-cell stride (band width / K)
        x = (w - s) / 2 = s * p / (2 * (1 - p))

    ``p = 0.25`` gives the 1/4-overlap / 1/2-alone / 1/4-overlap layout
    (``w = 4s/3``, ``x = s/6``; shared zone ``2x = w/4``; exclusive core
    ``s - 2x = w/2``). ``p < 0.5`` is REQUIRED: at 0.5 the exclusive core
    vanishes and a frequency could cover 3+ cells.

    Each interior edge's ``x`` uses the stride of the band CONTAINING that
    edge (bands may have unequal widths on the get_n grid; uniform grids get
    one value everywhere). The two END edges get ``x = 0``: the analysis
    window never widens, so neighbour indices stay in range by construction.

    Args:
        band_edges: 1D ascending sub-band edges (Hz).
        cap_edges: the cap-cell edge array from :func:`make_cap_edges`
            (nested or staggered) built over the same ``band_edges``.
        divisor: cap cells per sub-band (``K``).
        overlap_frac: ``p`` in ``[0, 0.5)``.

    Returns:
        ``np.ndarray`` of length ``len(cap_edges)`` -- one half-extension
        per cap edge, zeros at both ends.
    """
    p = float(overlap_frac)
    if not (0.0 <= p < 0.5):
        raise ValueError(
            f"cap overlap fraction (GB_CAP_OVERLAP_FRAC) must be in "
            f"[0, 0.5) -- at 0.5 the exclusive core vanishes and a leaf "
            f"could cover 3+ cells; got {p}."
        )
    be = np.asarray(band_edges, dtype=float)
    ce = np.asarray(cap_edges, dtype=float)
    step = (be[1:] - be[:-1]) / max(1, int(divisor))
    edge_band = np.clip(
        np.searchsorted(be, ce, side="right") - 1, 0, len(be) - 2
    )
    x = (p / (2.0 * (1.0 - p))) * step[edge_band]
    x[0] = 0.0
    x[-1] = 0.0
    return x


def cap_divisor_from_edges(band_edges, cap_edges) -> int:
    """Infer the cap divisor ``K`` from a stored (band_edges, cap_edges) pair."""
    nb = len(np.asarray(band_edges)) - 1
    nc = len(np.asarray(cap_edges)) - 1
    if nb <= 0 or nc <= 0 or nc % nb != 0:
        raise ValueError(
            f"cap grid ({nc} cells) is not an integer refinement of the band "
            f"grid ({nb} bands)."
        )
    return nc // nb


def _cap_grid_is_staggered(band_info: dict) -> bool:
    """Does the stored cap grid differ in VALUE from the band grid?

    The two are distinguishable only by their edge VALUES when the counts
    match: an unstaggered divisor-1 grid IS ``band_edges``, while the
    midpoint-to-midpoint grid has the same length with every interior edge
    shifted to a band midpoint. A count comparison cannot tell them apart,
    which is exactly the conflation :func:`ensure_cap_cell_fields`
    documents. Different counts (divisor >= 2) allocate regardless, so the
    answer only has to be right when the lengths are equal.
    """
    be = band_info.get("band_edges")
    ce = band_info.get("cap_edges")
    if be is None or ce is None:
        return False
    be = np.asarray(be, dtype=float)
    ce = np.asarray(ce, dtype=float)
    if be.shape != ce.shape:
        return False
    return not np.allclose(be, ce, rtol=0.0, atol=1e-12)


#: the per-WALKER cap family (``GB_LEAF_CAP_PER_WALKER``, user request
#: 2026-09-22). Each name is the ``_w`` twin of the shared 1-D array it
#: replaces as the GATE's state; the 1-D originals stay written as
#: max-over-walkers mirrors so the monitor and every diagnostic script keep
#: working. Deliberately SEPARATE NAMES rather than a reshaped array: a
#: store written with the feature on must still be readable by a build with
#: it off (and vice versa), which a changed rank on an existing key would
#: break. See :func:`ensure_cap_cell_fields`.
CAP_CELL_PER_WALKER_FIELDS = (
    ("cap_cell_leaf_cap_w", "cap_cell_leaf_cap"),
    ("cap_cell_iters_w", "cap_cell_iters"),
    ("cap_cell_best_ll_w", "cap_cell_best_ll"),
)


def ensure_cap_cell_fields(band_info: dict, num_cells: int,
                           staggered: bool = False,
                           per_walker: bool = False) -> None:
    """Backfill the per-CAP-CELL progressive leaf-cap arrays on ``band_info``.

    The cap-cell twins of the ``band_*`` cap arrays (see
    :func:`ensure_leaf_cap_fields`). These are the ARRAYS THE CAP MACHINERY
    ACTUALLY READS as of 2026-08-15; the ``band_leaf_cap`` family stays
    written (as the per-band max over its cells) purely so the monitor and
    the existing diagnostics keep working.

    - ``cap_cell_leaf_cap``: max alive leaves allowed per cap cell at EVERY
      temperature. ``-1`` = disarmed sentinel.
    - ``cap_cell_iters``: RJ iterations spent at the current cell cap.
    - ``cap_cell_best_ll``: running max of the per-cell cold-walker
      likelihood statistic at the current cap (reset on each increment).
    - ``cap_cell_cold_ll``: ``(nwalkers, num_cells)`` per-cold-walker
      statistic, refreshed every step so the decision is auditable.

    DIVISOR-1 SHORT CIRCUIT: when the cap grid IS the band grid there is
    nothing to allocate -- the move reads the ``band_*`` arrays directly, so
    divisor 1 is bit-identical to the pre-2026-08-15 code AND keeps
    resuming stores written before the cap grid existed.

    ``staggered`` DISABLES that short circuit (2026-08-29). The count
    ``num_cells == num_bands`` is NOT the same question as "the cap grid is
    the band grid": at ``cap_divisor == 1`` WITH stagger the counts are
    equal (1232 cells over 1232 sub-bands) while membership is shifted half
    a sub-band -- cell ``i`` runs midpoint-to-midpoint. Those arrays are
    then genuinely needed, and the move's ``_cap_state_arrays`` correctly
    asks for them (its predicate is ``_cap_is_band_grid`` =
    ``divisor == 1 and not stagger``). Keying this on the count instead
    skipped the allocation and the move raised ``KeyError`` on
    ``cap_cell_leaf_cap`` at construction.

    ``per_walker`` (``GB_LEAF_CAP_PER_WALKER``, 2026-09-22) additionally
    allocates the ``(nwalkers, num_cells)`` twins in
    :data:`CAP_CELL_PER_WALKER_FIELDS` -- the arrays the gate then reads
    and writes, with the 1-D originals demoted to max-over-walkers
    mirrors. Like ``staggered`` it DISABLES the divisor-1 short circuit:
    the shared arrays can stand in for the cell arrays when cells are
    bands, but ``band_leaf_cap`` has no walker axis to stand in for, so
    the cell family must exist at every divisor.

    ENABLING ON AN EXISTING STORE needs no migration. When the shared
    arrays are already present (any store written before this flag) each
    walker is SEEDED from them by broadcast, so every walker inherits the
    stored cap and its patience clock and divergence starts from the next
    increment. A fresh state seeds from the fresh sentinels instead, which
    is the same code path.

    OFF is the flag-off guarantee this whole design rests on: not one
    array is allocated and not one key is added, so a store written by a
    build carrying this change is byte-identical to one written without
    it. ``tests/test_gb_cap_per_walker.py`` asserts exactly that.
    """
    if (not staggered and not per_walker
            and int(num_cells) == int(band_info.get("num_bands", num_cells))):
        return
    band_info.setdefault("cap_cell_leaf_cap", np.full(num_cells, -1, dtype=int))
    band_info.setdefault("cap_cell_iters", np.zeros(num_cells, dtype=int))
    band_info.setdefault("cap_cell_best_ll", np.full(num_cells, -np.inf))
    _nw = int(band_info.get("nwalkers", 0) or 0)
    if _nw:
        band_info.setdefault(
            "cap_cell_cold_ll", np.full((_nw, num_cells), -np.inf)
        )
    if per_walker:
        if not _nw:
            raise ValueError(
                "GB_LEAF_CAP_PER_WALKER needs band_info['nwalkers'] to size "
                "the per-walker cap arrays, but it is absent or zero. A "
                "band_info that reached here without a walker count is a "
                "half-built state -- refusing rather than silently falling "
                "back to the shared cap, which would look identical in the "
                "log and gate every walker on one allowance."
            )
        for _name, _shared in CAP_CELL_PER_WALKER_FIELDS:
            if _name in band_info and band_info[_name] is not None:
                continue
            # SEED by broadcast from the shared array, which at this point
            # is either the freshly-allocated sentinel fill or the value
            # restored from a store written before the flag existed.
            band_info[_name] = np.repeat(
                np.asarray(band_info[_shared])[None, :], _nw, axis=0
            )


#: the per-band RJ shutoff valve's persisted state. ALL-OR-NOTHING: the
#: streak, the previous occupancy it is measured against, the shut-off set
#: and the two revival counters are one consistent record, so a partial or
#: mismatched set is discarded whole rather than half-honoured.
BAND_SHUTOFF_FIELDS = (
    "band_occ_streak",
    "band_occ_last",
    "band_rj_shutoff",
    "band_shutoff_since_revive",
    "band_shutoff_epoch",
)

#: sentinel for "no F-stat epoch recorded yet" in ``band_shutoff_epoch``
BAND_SHUTOFF_EPOCH_UNSET = -1


def _zero_band_shutoff(band_info: dict, num_bands: int) -> None:
    """Install a fresh (zeroed) shutoff record on ``band_info``."""
    band_info["band_occ_streak"] = np.zeros(num_bands, dtype=np.int64)
    # -1, not 0: the streak only counts an iteration whose occupancy is
    # UNCHANGED, and -1 is unreachable for a count, so the next update
    # always starts a fresh streak at 1. Same rule as _band_shutoff_revive.
    band_info["band_occ_last"] = np.full(num_bands, -1, dtype=np.int64)
    band_info["band_rj_shutoff"] = np.zeros(num_bands, dtype=bool)
    band_info["band_shutoff_since_revive"] = np.zeros(1, dtype=np.int64)
    band_info["band_shutoff_epoch"] = np.full(
        1, BAND_SHUTOFF_EPOCH_UNSET, dtype=np.int64)


def ensure_band_shutoff_fields(band_info: dict, num_bands: int) -> str:
    """Backfill/validate the per-band RJ shutoff valve state.

    USER PROPOSAL 2026-08-29. The valve's counters used to be plain
    per-process memory (allocated under ``if not hasattr(self, ...)`` on
    the move), so every process restart wiped the clock. gf_prod_3mo_v7
    took 26 launches with segments of 2-8 iterations against a 5-tick
    clock, so on a spot-preempted cluster the valve would barely work even
    though the call site is now fixed. Persisting the record here makes
    the clock count GB proposes across the WHOLE run.

    Rides the SAME channel as the ``band_leaf_cap`` family and needs no
    schema change: ``GBState.storage_arrays`` persists every ndarray in
    ``band_info``, and ``GBState.from_stored`` restores every ``band_*``
    key, so naming is the whole wiring. Deliberately kept OUT of
    ``band_info_keys`` (like the leaf-cap family) so band-info dicts from
    older stores still pass the setter's required-key check, and out of
    ``legacy_dtype_names`` so the counters keep int64/bool rather than
    being coerced to the backend float dtype.

    Returns a short origin token for the status line -- ``"fresh"``,
    ``"restored"``, or ``"reset(...)"`` -- because a persisted clock that
    silently failed to restore would be the same invisible failure this
    whole investigation was about.

    Degrades, never raises:

    * ABSENT (any store written before this existed) -> fresh zeros.
    * PARTIAL (a half-written record) -> discarded whole, fresh zeros.
    * LENGTH MISMATCH (the band grid changed between runs) -> discarded,
      fresh zeros, warning naming both lengths. Restoring a streak
      indexed by a different grid would freeze arbitrary bands.
    """
    present = [f for f in BAND_SHUTOFF_FIELDS if band_info.get(f) is not None]
    if not present:
        _zero_band_shutoff(band_info, num_bands)
        return "fresh"
    if len(present) != len(BAND_SHUTOFF_FIELDS):
        missing = [f for f in BAND_SHUTOFF_FIELDS if f not in present]
        logger.warning(
            "band shutoff state is incomplete (missing %s); discarding the "
            "partial record and starting the valve's clock from zero.",
            missing)
        _zero_band_shutoff(band_info, num_bands)
        return "reset(partial)"
    for name in ("band_occ_streak", "band_occ_last", "band_rj_shutoff"):
        stored = int(np.shape(band_info[name])[-1])
        if stored != int(num_bands):
            logger.warning(
                "stored band shutoff state %r covers %d bands but this "
                "run's grid has %d; the band grid changed between runs, so "
                "the valve's clock is being restarted from zero rather than "
                "restored onto a grid it was not measured on.",
                name, stored, int(num_bands))
            _zero_band_shutoff(band_info, num_bands)
            return f"reset(grid {stored}!={int(num_bands)})"
    # Restored: pin the dtypes. An HDF5 round trip is faithful for these
    # (they are not in legacy_dtype_names) but a hand-built or migrated
    # band_info need not be, and the streak arithmetic is integer.
    band_info["band_occ_streak"] = np.asarray(
        band_info["band_occ_streak"], dtype=np.int64)
    band_info["band_occ_last"] = np.asarray(
        band_info["band_occ_last"], dtype=np.int64)
    band_info["band_rj_shutoff"] = np.asarray(
        band_info["band_rj_shutoff"], dtype=bool)
    for name in ("band_shutoff_since_revive", "band_shutoff_epoch"):
        band_info[name] = np.asarray(
            band_info[name], dtype=np.int64).reshape(-1)[:1]
    return "restored"


#: the per-(walker, band) SEARCH STAGE record
#: (``GB_SEARCH_STAGE_PER_WALKER``, user request 2026-09-23). ALL-OR-NOTHING
#: for exactly the reason :data:`BAND_SHUTOFF_FIELDS` is: the stage, the
#: occupancy it was measured against and the streak that earned it are one
#: consistent record, and half of it restored onto the other half's grid
#: would promote arbitrary (walker, band) pairs to the relaxed floor.
SEARCH_STAGE_FIELDS = (
    "band_stage_w",
    "band_stage_occ_last_w",
    "band_stage_streak_w",
)

#: stage indices. One-way: COARSE -> FINE, never back (see
#: ``GBSpecialBase._update_search_stages`` for why demotion is unsafe).
SEARCH_STAGE_COARSE = 0
SEARCH_STAGE_FINE = 1


def _zero_search_stage(band_info: dict, num_bands: int, nwalkers: int) -> None:
    """Install a fresh (COARSE everywhere) search-stage record."""
    band_info["band_stage_w"] = np.zeros(
        (nwalkers, num_bands), dtype=np.int8)
    # -1, not 0: the streak only counts an iteration whose occupancy is
    # UNCHANGED, and -1 is unreachable for a count, so the first update
    # always starts a fresh streak rather than crediting a spurious match
    # against a zero-filled array. Same sentinel, same reason, as
    # ``band_occ_last`` in _zero_band_shutoff.
    band_info["band_stage_occ_last_w"] = np.full(
        (nwalkers, num_bands), -1, dtype=np.int64)
    band_info["band_stage_streak_w"] = np.zeros(
        (nwalkers, num_bands), dtype=np.int64)
    # 1-D MIRROR, min over walkers: the monitor and the diagnostic scripts
    # get one number per band, and a band only reads FINE once EVERY walker
    # agrees. Like the cap family's 1-D mirrors this is a summary view and
    # never the latch's state.
    band_info["band_stage"] = np.zeros(num_bands, dtype=np.int8)


def ensure_search_stage_fields(band_info: dict, num_bands: int,
                               per_walker: bool = False) -> str:
    """Backfill/validate the per-(walker, band) search-stage record.

    The search's opt-SNR floor (and, when a later change picks it up, its
    F-stat peak floor) is a function of a ``(walker, band)`` STAGE rather
    than one number for the whole run: stage ``0`` = COARSE (the starting
    floors, hunting bright sources), stage ``1`` = FINE (relaxed floors,
    digging the faint tail). A band promotes once that walker's cold-chain
    SOURCE COUNT there has stopped changing -- see
    ``GBSpecialBase._update_search_stages``.

    Rides the SAME channel as the ``band_leaf_cap`` and shutoff families and
    needs no schema change: ``GBState.storage_arrays`` persists every
    ndarray in ``band_info`` and ``GBState.from_stored`` restores every
    ``band_*`` key, so naming is the whole wiring. Deliberately kept OUT of
    ``band_info_keys`` (so band-info dicts from older stores still pass the
    setter's required-key check) and out of ``legacy_dtype_names`` (these
    are integer counters and a stage index; coercing them to the backend
    float dtype, which is what that list is for, would be wrong -- the same
    reasoning the shutoff family follows).

    ``per_walker`` FALSE is a hard no-op: not one array is allocated and not
    one key is added, so a store written by a build carrying this change is
    byte-identical to one written without it.

    Returns a short origin token for the status line -- ``"off"``,
    ``"fresh"``, ``"restored"`` or ``"reset(...)"`` -- because a persisted
    latch that silently failed to restore is the same invisible failure the
    shutoff valve's own clock had.

    Degrades, never raises:

    * ABSENT (any store written before this existed) -> fresh COARSE.
    * PARTIAL (a half-written record) -> discarded whole, fresh COARSE.
    * SHAPE MISMATCH (the band grid or the walker count changed between
      runs) -> discarded, fresh COARSE, warning naming both shapes.

    Every degradation lands on COARSE, which is the TIGHTER floor: a
    record we cannot trust must never hand a walker the relaxed prior it
    did not earn.
    """
    if not per_walker:
        return "off"
    _nw = int(band_info.get("nwalkers", 0) or 0)
    if not _nw:
        raise ValueError(
            "GB_SEARCH_STAGE_PER_WALKER needs band_info['nwalkers'] to size "
            "the per-walker stage arrays, but it is absent or zero. A "
            "band_info that reached here without a walker count is a "
            "half-built state -- refusing rather than silently falling back "
            "to one shared stage, which would look identical in the log and "
            "move every walker's floor together."
        )
    present = [f for f in SEARCH_STAGE_FIELDS if band_info.get(f) is not None]
    if not present:
        _zero_search_stage(band_info, num_bands, _nw)
        return "fresh"
    if len(present) != len(SEARCH_STAGE_FIELDS):
        missing = [f for f in SEARCH_STAGE_FIELDS if f not in present]
        logger.warning(
            "search-stage state is incomplete (missing %s); discarding the "
            "partial record and restarting every (walker, band) at the "
            "COARSE floor.", missing)
        _zero_search_stage(band_info, num_bands, _nw)
        return "reset(partial)"
    _want = (_nw, int(num_bands))
    for name in SEARCH_STAGE_FIELDS:
        got = tuple(np.shape(band_info[name]))
        if got != _want:
            logger.warning(
                "stored search-stage state %r has shape %s but this run's "
                "grid is %s; the band grid or the walker count changed "
                "between runs, so the stage latch is being restarted at the "
                "COARSE floor rather than restored onto a grid it was not "
                "measured on.", name, got, _want)
            _zero_search_stage(band_info, num_bands, _nw)
            return f"reset(shape {got}!={_want})"
    # Restored: pin the dtypes. An HDF5 round trip is faithful for these
    # (they are not in legacy_dtype_names) but a hand-built or migrated
    # band_info need not be, and the streak arithmetic is integer.
    band_info["band_stage_w"] = np.asarray(
        band_info["band_stage_w"], dtype=np.int8)
    band_info["band_stage_occ_last_w"] = np.asarray(
        band_info["band_stage_occ_last_w"], dtype=np.int64)
    band_info["band_stage_streak_w"] = np.asarray(
        band_info["band_stage_streak_w"], dtype=np.int64)
    band_info["band_stage"] = np.asarray(
        band_info["band_stage_w"], dtype=np.int8).min(axis=0)
    return "restored"


#: the per-(walker, band) RJ shutoff valve
#: (``GB_SEARCH_BAND_SHUTOFF_PER_WALKER``, user request 2026-09-24). This is
#: ADDITIONAL to, and completely independent of, the per-band valve in
#: :data:`BAND_SHUTOFF_FIELDS` -- that one stays exactly as it is and the two
#: compose with OR (a row is frozen if either valve says so).
SEARCH_SHUTOFF_FIELDS = (
    "band_rj_shutoff_w",
    "band_shutoff_w_step",
)

#: The valve's WINDOW -- the running-best lnL and the consecutive
#: non-improving count that earn the boolean above (user request
#: 2026-09-27). Held SEPARATE from :data:`SEARCH_SHUTOFF_FIELDS` on purpose:
#: those two are an all-or-nothing group, so listing the window there would
#: make every store written before this existed fail the completeness check
#: and reopen a valve it had legitimately earned. Here the window is
#: backfilled on its own, and the boolean survives.
#:
#: WHY IT IS PERSISTED AT ALL. It used to live only in
#: ``GBSpecialBase._shutoff_best`` / ``._shutoff_streak``, so a restart
#: re-earned the whole window. With ``search_shutoff_conv_iter = 3`` and the
#: running best starting at -inf, the first iteration after a restart always
#: counts as an improvement, so FOUR consecutive iterations inside ONE job
#: were needed before any pair could shut. Measured on the 6mo v9 run at
#: ~1.8 h/iteration: job 643 (8.9 h) shut 1478 -> 1527 pairs; jobs 646
#: (2.1 h), 648 (17 min) and 650 (3.2 h) shut ZERO, and 650's valve reported
#: "inert" twice. The gate could not close because the jobs were shorter
#: than the window, not because the bands were still paying.
SEARCH_SHUTOFF_WINDOW_FIELDS = (
    "band_shutoff_best_w",
    "band_shutoff_streak_w",
    "band_cold_logl_max_w",
    "band_cold_logl_w",
    "band_shutoff_reset_w",
    "band_cold_logl_peak_w",
)

#: ``band_shutoff_reset_w`` counts how many times each (walker, sub-band)
#: has had its shutoff streak ZEROED -- i.e. how often it beat its
#: all-time cold-lnL max by more than the tolerance (user ruling
#: 2026-09-27: "they are resetting? we need to diagnose that").
#:
#: A pair can be unshut for three quite different reasons and the streak
#: alone cannot tell them apart:
#:   * it keeps resetting        -> high reset count, streak near 0
#:   * it is simply still young  -> reset count 0, streak 1 or 2
#:   * it was only just occupied -> reset count 0, streak 0
#: Persisted with the rest so a snapshot answers this directly instead of
#: leaving it to be inferred from the shut TOTAL.

#: ``band_cold_logl_w`` is THIS iteration's per-(walker, band) cold-chain
#: lnL -- the raw statistic the valve judges, recorded so the store keeps a
#: HISTORY of it (user ruling 2026-09-27: "we should track this better").
#:
#: WHY IT DID NOT EXIST. The valve computes the statistic fresh every
#: iteration through ``_cap_stats_local`` and discards it. The only array
#: that ever held a per-band cold lnL is ``band_cold_ll``, and that one is
#: written by the leaf-CAP gate -- which this schedule disarms. Measured on
#: 6mo v9 job 650: ``band_cold_ll`` and ``cap_cell_cold_ll`` are ``-inf``
#: in ALL 4,928 cells of every stored row. So the quantity that decides
#: when a search stage ends had no recorded history at all, which is why
#: "have the sub-band lnLs converged?" could not be answered from a
#: snapshot.
#:
#: With this, the three arrays together tell the whole story per cell:
#: the value now, the all-time max it is judged against, and how many
#: consecutive iterations it has failed to beat that max.
#:
#: UNITS (user ruling 2026-10-10): the statistic -- and so
#: ``band_cold_logl_w`` / ``_max_w`` / ``_peak_w`` -- is the band's GAIN
#: ``1/2 <d|d> - 1/2 <r|r>`` under the walker's current noise, no longer the
#: residual term ``-1/2 <r|r>``. Rows written before hold residual units
#: (~-1,400 per band against a gain >= ~0); the first judge after the change
#: resets each streak once (``GBSpecialBase._update_search_band_shutoff``).

#: ``band_cold_logl_max_w`` is DIFFERENT IN KIND from the two above and is
#: the reference the valve actually judges against (user ruling
#: 2026-09-27: "you should hold an absolute logL max value for the cold
#: chain per (band, walker); it should persist through everything
#: including saving. We want to monitor against that. Not a simple
#: increase.").
#:
#: ABSOLUTE and NEVER RELEASED. ``band_shutoff_best_w`` and
#: ``band_shutoff_streak_w`` are the per-STEP window and are cleared
#: whenever the recipe step changes; this one is the all-time maximum
#: cold-chain lnL that (walker, band) has ever reached, and it survives a
#: step change, a restart and a save. That is the whole point: a
#: reference that resets cannot tell "this band is fitted" from "this
#: band's evidence was just thrown away", and the resetting reference is
#: why the valve spent jobs 646/648/650 unable to latch.
#:
#: ⚠ IT REVERSES the 2026-09-24 note on ``_release_search_band_shutoff``
#: ("a step that inherited the previous step's running-best lnL would
#: re-freeze on its first patience window"). That risk is real and is now
#: accepted deliberately: if a new step's moves genuinely find something
#: better they beat the all-time max and the streak resets, and if they
#: cannot, the band is finished and freezing it is correct.
SEARCH_SHUTOFF_ABSOLUTE_FIELD = "band_cold_logl_max_w"

#: sentinel for "no recipe step recorded yet" in ``band_shutoff_w_step``
SEARCH_SHUTOFF_STEP_UNSET = -1

#: THE VALVE'S WINDOW RULE (user ruling 2026-10-13: "Give every sub-band its
#: covering layer as the valve window"). The valve's gain is formed on a
#: window of whole WDM layers (``GBSpecialBase._window_bins``), and the rule
#: that picks those layers is versioned here, stamped per store in
#: ``band_info["band_window_rule"]`` (shape ``(1,)``, int64), so a record
#: learned under one rule is never judged as if it had been learned under
#: another.
#:
#: * ``1`` -- CENTRES (every store written before 2026-10-13): the layers
#:   whose CENTRE ``m * layer_df`` lies inside the band. The production
#:   sub-bands are 1/8 of a layer wide, so 924 of the 1232 9mo sub-bands held
#:   no centre, had an EMPTY window and a gain of exactly 0 on every row, and
#:   their occupied pairs shut after the patience count whatever their
#:   sources did (9mo row 66, the first gb_search_2 row with a shut pair:
#:   40.6 % of all blind pairs shut -- about every occupied one -- against
#:   7.7 % of the window-owning pairs).
#: * ``2`` -- COVERING: every band owns the layer(s) whose WDM tile
#:   ``[(m - 1/2), (m + 1/2)) * layer_df`` it overlaps. A superset of rule 1
#:   (identical wherever the centre nearest each band edge lies inside the
#:   band: the 308 window-owning 9mo sub-bands keep their layer), never
#:   empty.
#:
#: The fit's first judge under a newer rule than the stamp RE-LEARNS the
#: running max, peak, best and streak of every (walker, band) once and
#: re-stamps (``gbspecialstretch.relearn_valve_window``); the stamp is saved
#: with the rest of ``band_info``, so a resume never repeats it.
VALVE_WINDOW_RULE_FIELD = "band_window_rule"
VALVE_WINDOW_RULE_CENTRES = 1
VALVE_WINDOW_RULE_COVERING = 2
#: the rule this code judges under
VALVE_WINDOW_RULE = VALVE_WINDOW_RULE_COVERING


def ensure_valve_window_rule(band_info: dict) -> int:
    """Make sure ``band_info`` carries the valve's window-rule stamp.

    A stored stamp is kept (dtype and shape pinned). An ABSENT stamp is
    inferred from the record it would describe: a window that holds
    anything learned (a finite running max, best or peak, or a non-zero
    streak) was learned by code older than the stamp, i.e. under
    :data:`VALVE_WINDOW_RULE_CENTRES`; an empty one (a fresh valve, or one
    that never judged) has nothing to re-learn and is stamped with the
    current rule. Returns the rule now on ``band_info``.
    """
    cur = band_info.get(VALVE_WINDOW_RULE_FIELD)
    if cur is not None:
        arr = np.asarray(cur, dtype=np.int64).reshape(-1)[:1]
        if arr.size == 1:
            band_info[VALVE_WINDOW_RULE_FIELD] = arr
            return int(arr[0])
    learned = False
    for name in ("band_cold_logl_max_w", "band_shutoff_best_w",
                 "band_cold_logl_peak_w"):
        a = band_info.get(name)
        if a is not None and np.isfinite(np.asarray(a, dtype=float)).any():
            learned = True
    a = band_info.get("band_shutoff_streak_w")
    if a is not None and np.any(np.asarray(a) != 0):
        learned = True
    rule = VALVE_WINDOW_RULE_CENTRES if learned else VALVE_WINDOW_RULE
    band_info[VALVE_WINDOW_RULE_FIELD] = np.full(1, rule, dtype=np.int64)
    return rule


def _zero_search_shutoff_window(band_info: dict, num_bands: int,
                                nwalkers: int) -> None:
    """Install a fresh (empty) shutoff window: no best, no streak.

    ``-inf`` for the best is the load-bearing value -- it is what makes the
    first measured lnL count as an improvement, so a fresh window can never
    shut a band on its first iteration.
    """
    band_info["band_shutoff_best_w"] = np.full(
        (nwalkers, num_bands), -np.inf, dtype=np.float64)
    band_info["band_shutoff_streak_w"] = np.zeros(
        (nwalkers, num_bands), dtype=np.int64)
    # NOT cleared by a release -- see SEARCH_SHUTOFF_ABSOLUTE_FIELD. A
    # release goes through ``_release_search_band_shutoff``, which never
    # calls this. What DOES reach here is a fresh install or a grid
    # change, and in both cases an existing absolute max is meaningless:
    # there is no history on a fresh grid, and a max measured on a
    # different band grid belongs to different bands.
    #
    # Shape-checked rather than ``setdefault``: this function is the
    # shape-mismatch recovery path, so a plain setdefault would leave the
    # very array that failed validation in place.
    _abs = band_info.get("band_cold_logl_max_w")
    if _abs is None or np.shape(_abs) != (int(nwalkers), int(num_bands)):
        band_info["band_cold_logl_max_w"] = np.full(
            (nwalkers, num_bands), -np.inf, dtype=np.float64)
    # The per-iteration value is pure telemetry -- nothing reads it back,
    # so it is always installed fresh rather than preserved.
    band_info["band_cold_logl_w"] = np.full(
        (nwalkers, num_bands), -np.inf, dtype=np.float64)
    # Reset COUNTS ride with the window: a release restarts the question,
    # so the count is per-step like the streak, not all-time like the max.
    band_info["band_shutoff_reset_w"] = np.zeros(
        (nwalkers, num_bands), dtype=np.int64)
    # WITHIN-ITERATION peak, accumulated at the end of every in-model
    # repeat group and consumed (and reset) once per iteration by the
    # gate. -inf so the first observe of a cycle always wins.
    band_info["band_cold_logl_peak_w"] = np.full(
        (nwalkers, num_bands), -np.inf, dtype=np.float64)


def _zero_search_shutoff(band_info: dict, num_bands: int, nwalkers: int,
                         step: int = SEARCH_SHUTOFF_STEP_UNSET) -> None:
    """Install a fresh (all-open) per-(walker, band) RJ shutoff record."""
    band_info["band_rj_shutoff_w"] = np.zeros(
        (nwalkers, num_bands), dtype=bool)
    band_info["band_shutoff_w_step"] = np.full(1, int(step), dtype=np.int64)
    # The window goes with it. Reopening the valve while keeping a running
    # best measured under the old configuration is the exact failure
    # ``_release_search_band_shutoff`` documents: the model is already
    # fitted, so nothing beats the inherited best and every pair re-freezes
    # on its first patience window, which reads as "converged" and is not.
    _zero_search_shutoff_window(band_info, num_bands, nwalkers)


def ensure_search_shutoff_window(band_info: dict, num_bands: int,
                                 nwalkers: int) -> str:
    """Backfill/validate the shutoff window alongside the valve.

    Independent of the valve's own completeness check so that a store
    written before 2026-09-27 -- which has the boolean and the step stamp
    but neither window array -- KEEPS its earned valve and simply starts
    the window empty. That is the same degradation the in-memory window
    always had, so this can only improve on it.
    """
    present = [f for f in SEARCH_SHUTOFF_WINDOW_FIELDS
               if band_info.get(f) is not None]
    _want = (int(nwalkers), int(num_bands))
    if len(present) != len(SEARCH_SHUTOFF_WINDOW_FIELDS):
        _zero_search_shutoff_window(band_info, num_bands, nwalkers)
        return "fresh" if not present else "reset(partial)"
    for name in SEARCH_SHUTOFF_WINDOW_FIELDS:
        if tuple(np.shape(band_info[name])) != _want:
            logger.warning(
                "stored RJ shutoff window %r has shape %s but this run's "
                "grid is %s; restarting the window rather than measuring a "
                "patience count against a grid it was not earned on.",
                name, tuple(np.shape(band_info[name])), _want)
            _zero_search_shutoff_window(band_info, num_bands, nwalkers)
            return "reset(shape)"
    # Pin the dtypes. The HDF round trip is faithful (neither name is in
    # ``legacy_dtype_names``, so they keep float64/int64 rather than being
    # cast to the backend float), but a migrated or hand-built band_info
    # need not be, and the streak arithmetic is integer.
    band_info["band_shutoff_best_w"] = np.asarray(
        band_info["band_shutoff_best_w"], dtype=np.float64)
    band_info["band_shutoff_streak_w"] = np.asarray(
        band_info["band_shutoff_streak_w"], dtype=np.int64)
    band_info["band_cold_logl_max_w"] = np.asarray(
        band_info["band_cold_logl_max_w"], dtype=np.float64)
    band_info["band_cold_logl_w"] = np.asarray(
        band_info["band_cold_logl_w"], dtype=np.float64)
    band_info["band_shutoff_reset_w"] = np.asarray(
        band_info["band_shutoff_reset_w"], dtype=np.int64)
    band_info["band_cold_logl_peak_w"] = np.asarray(
        band_info["band_cold_logl_peak_w"], dtype=np.float64)
    return "restored"


def ensure_search_shutoff_fields(band_info: dict, num_bands: int,
                                 per_walker: bool = False) -> str:
    """Backfill/validate the per-(walker, band) RJ shutoff valve.

    A SEARCH-only valve, additional to the existing per-band one: once a
    (walker, band)'s cold-chain source count has stopped growing WITHIN THE
    CURRENT RECIPE STEP, that band takes no further RJ on that walker until
    the next recipe step begins. See
    ``GBSpecialBase._update_search_band_shutoff`` for the criterion, which
    is the per-(walker, band) form of ``RJRecipeStep.stopping_function``'s
    nleaves-max window test.

    ``band_shutoff_w_step`` records WHICH recipe step the valve was earned
    in. The move stamps the step serial it is running under; a mismatch on
    resume means the run advanced a step while it was down, and the whole
    valve is released -- the alternative, honouring a valve earned in a
    step that is over, would silently freeze bands the new step was meant
    to reopen.

    ``per_walker`` FALSE is a hard no-op. Same all-or-nothing degradation
    rules as :func:`ensure_band_shutoff_fields`, and every degradation
    lands on ALL-OPEN, which is the permissive direction: a valve we
    cannot trust must never freeze a band the search still needs.
    """
    if not per_walker:
        return "off"
    _nw = int(band_info.get("nwalkers", 0) or 0)
    if not _nw:
        raise ValueError(
            "GB_SEARCH_BAND_SHUTOFF_PER_WALKER needs band_info['nwalkers'] "
            "to size the per-walker shutoff valve, but it is absent or "
            "zero. A band_info that reached here without a walker count is "
            "a half-built state -- refusing rather than silently falling "
            "back to the per-band valve, which would look identical in the "
            "log and freeze every walker together."
        )
    present = [f for f in SEARCH_SHUTOFF_FIELDS
               if band_info.get(f) is not None]
    # Every return below carries the window-rule stamp (see
    # VALVE_WINDOW_RULE): judged after the valve and its window are in
    # place, because an absent stamp is inferred from what the window holds.
    if not present:
        _zero_search_shutoff(band_info, num_bands, _nw)
        ensure_valve_window_rule(band_info)
        return "fresh"
    if len(present) != len(SEARCH_SHUTOFF_FIELDS):
        missing = [f for f in SEARCH_SHUTOFF_FIELDS if f not in present]
        logger.warning(
            "per-walker RJ shutoff state is incomplete (missing %s); "
            "discarding the partial record and reopening every "
            "(walker, band).", missing)
        _zero_search_shutoff(band_info, num_bands, _nw)
        ensure_valve_window_rule(band_info)
        return "reset(partial)"
    _want = (_nw, int(num_bands))
    got = tuple(np.shape(band_info["band_rj_shutoff_w"]))
    if got != _want:
        logger.warning(
            "stored per-walker RJ shutoff state has shape %s but this "
            "run's grid is %s; the band grid or the walker count changed "
            "between runs, so the valve is being reopened rather than "
            "restored onto a grid it was not measured on.", got, _want)
        _zero_search_shutoff(band_info, num_bands, _nw)
        ensure_valve_window_rule(band_info)
        return "reset(shape)"
    band_info["band_rj_shutoff_w"] = np.asarray(
        band_info["band_rj_shutoff_w"], dtype=bool)
    band_info["band_shutoff_w_step"] = np.asarray(
        band_info["band_shutoff_w_step"], dtype=np.int64).reshape(-1)[:1]
    _win = ensure_search_shutoff_window(band_info, num_bands, _nw)
    ensure_valve_window_rule(band_info)
    if _win != "restored":
        # The valve survives; only its window restarts. Say which, because
        # "restored" on the valve with a fresh window means the run has to
        # re-earn `search_shutoff_conv_iter` iterations before anything NEW
        # can shut -- worth seeing in the log rather than inferring from a
        # stalled counter.
        return f"restored (window {_win})"
    return "restored"


def ensure_leaf_cap_fields(band_info: dict, num_bands: int,
                           per_walker: bool = False) -> None:
    """Backfill the per-band progressive leaf-cap arrays on ``band_info``.

    Three ``(num_bands,)`` arrays drive the search-mode leaf cap
    (see ``GBSpecialBase._update_band_leaf_caps``):

    - ``band_leaf_cap``: max alive leaves allowed per band at EVERY
      temperature. ``-1`` = cap disarmed (the fresh-state sentinel; the
      first cap-enabled RJ move arms it to its ``leaf_cap_start``).
    - ``band_cap_iters``: RJ iterations spent at the current cap.
    - ``band_best_ll``: running max of the per-band cold-walker residual
      ll at the current cap (reset to ``-inf`` on each increment).

    Kept OUT of ``band_info_keys`` so band-info dicts loaded from HDF5
    files written before this feature still pass the setter's required-key
    check; this backfill runs from both branches of
    :meth:`GBState.initialize_band_information`.
    """
    band_info.setdefault("band_leaf_cap", np.full(num_bands, -1, dtype=int))
    band_info.setdefault("band_cap_iters", np.zeros(num_bands, dtype=int))
    band_info.setdefault("band_best_ll", np.full(num_bands, -np.inf))
    # ``band_cold_ll``: the per-band residual ll of EVERY cold-chain walker,
    # refreshed each step. ``band_best_ll`` is the running max over this;
    # keeping the full per-walker array makes the cap decision auditable
    # after the fact (and is what the lnL-improvement criterion is judged
    # from). ``(nwalkers, num_bands)``.
    _nw = int(band_info.get("nwalkers", 0) or 0)
    if _nw:
        band_info.setdefault(
            "band_cold_ll", np.full((_nw, num_bands), -np.inf)
        )
        if per_walker:
            # ``band_best_ll_w``: the per-WALKER running max, the twin of
            # ``band_best_ll``. Pure monitor series (no reset -- band caps
            # are a mirror on cell grids and never gate), and the only way
            # a snapshot can show whether the per-walker caps actually
            # diverged: ``band_best_ll`` is a max over walkers and so is
            # blind to exactly the spread this feature exists to expose.
            band_info.setdefault(
                "band_best_ll_w", np.full((_nw, num_bands), -np.inf)
            )


class ModuleSubState(eryn_State):
    """Base class for the per-module (per-branch) global-fit sub-states.

    A sub-state owns the module-specific sampler information for one branch
    (temperature ladders, per-band counters, ...). Its storage contract is
    "the sub-state IS the schema": the matching
    :class:`~lisatools.globalfit.hdfbackend.ModuleSubBackend` derives every
    HDF5 dataset from the arrays the sub-state allocates -- there is no
    separate schema layer. Subclasses implement/extend small setup methods
    and name lists:

    - :meth:`storage_arrays`: ``{on-disk name: array}`` written every saved
      iteration.
    - :meth:`static_arrays`: ``{on-disk name: array}`` written once at
      backend reset (e.g. ``band_edges``).
    - :meth:`storage_attrs`: ``{name: scalar}`` written as HDF5 group attrs
      at reset (e.g. ``num_bands``).
    - ``static_names`` / ``dim_attr_names``: name lists the backend uses to
      read ``reset_kwargs`` back from an existing file.
    - :meth:`make_template`: allocate a zeroed instance from dimension
      kwargs; the backend's ``reset`` shapes every dataset from it.
    - :meth:`from_stored`: rebuild an instance from one stored iteration.
    - :meth:`reset_delta_counters`: zero per-iteration-delta counters after
      each save (default: nothing to zero).
    """

    static_names: tuple = ()
    dim_attr_names: tuple = ("ntemps", "nwalkers", "nleaves_max", "ndim")

    #: arrays carried by the tempered block (allocated by
    #: :meth:`initialize_tempered`, copied by the copy path; missing names
    #: are skipped so subclasses with their own ladder storage compose)
    tempered_array_names: tuple = (
        "coords",
        "inds",
        "log_like",
        "log_prior",
        "betas",
        "d_h",
        "h_h",
        "in_model_proposed",
        "in_model_accepted",
        "rj_proposed",
        "rj_accepted",
        "swaps_proposed",
        "swaps_accepted",
    )
    #: name of the ladder array this class stores. The base allocates and
    #: persists a flat ``betas (ntemps,)``; subclasses with richer ladders
    #: (``band_temps``, ``betas_all``) override this and skip the flat one.
    betas_attr_name: str = "betas"
    #: counters zeroed after every save (per-iteration deltas)
    delta_counter_names: tuple = (
        "in_model_proposed",
        "in_model_accepted",
        "rj_proposed",
        "rj_accepted",
        "swaps_proposed",
        "swaps_accepted",
    )
    #: arrays stored with the backend's float dtype for continuity with the
    #: pre-rework files (everything else keeps its in-memory dtype)
    legacy_dtype_names: tuple = ()

    def __init__(self, possible_state=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            self._copy_tempered_from(possible_state, deepcopy if copy else return_x)

    # ------------------------------------------------------------------
    # Tempered ensemble block: the module's full (ntemps, nwalkers, ...)
    # ensemble, owned by the sub-state
    # ------------------------------------------------------------------

    @property
    def tempered_initialized(self) -> bool:
        """Whether the tempered ensemble block has been allocated."""
        return getattr(self, "_tempered_initialized", False)

    def _ll_shape(self):
        return (self.ntemps, self.nwalkers)

    def _counter_shape(self):
        return (self.ntemps,)

    def _swaps_shape(self):
        return (max(self.ntemps - 1, 0),)

    def initialize_tempered(self, ntemps, nwalkers, nleaves_max, ndim, coords=None, inds=None):
        """Allocate (or validate + refill) the module's tempered ensemble.

        Idempotent: on an already-initialized sub-state the geometry must
        match exactly and any provided ``coords`` / ``inds`` are copied in.
        """
        dims = (int(ntemps), int(nwalkers), int(nleaves_max), int(ndim))
        if self.tempered_initialized:
            current = (self.ntemps, self.nwalkers, self.nleaves_max, self.ndim)
            if current != dims:
                raise ValueError(
                    f"tempered geometry mismatch: sub-state has "
                    f"(ntemps, nwalkers, nleaves_max, ndim)={current}, "
                    f"initialize_tempered got {dims}."
                )
            if coords is not None:
                self.coords[:] = coords
            if inds is not None:
                self.inds[:] = inds
            return

        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = dims
        shape = dims[:1] + dims[1:2] + (self.nleaves_max, self.ndim)
        if coords is not None:
            coords = np.array(coords, dtype=float, copy=True)
            if coords.shape != shape:
                raise ValueError(
                    f"coords shape {coords.shape} does not match tempered "
                    f"geometry {shape}."
                )
            self.coords = coords
        else:
            self.coords = np.zeros(shape)
        if inds is not None:
            inds = np.array(inds, dtype=bool, copy=True)
            if inds.shape != shape[:-1]:
                raise ValueError(
                    f"inds shape {inds.shape} does not match tempered "
                    f"geometry {shape[:-1]}."
                )
            self.inds = inds
        else:
            self.inds = np.ones(shape[:-1], dtype=bool)

        self.log_like = np.zeros(self._ll_shape())
        self.log_prior = np.zeros(self._ll_shape())
        if self.betas_attr_name == "betas" and getattr(self, "betas", None) is None:
            self.betas = np.ones(self.ntemps)
        # per-leaf cold-chain inner products <d|h> and <h|h>, recorded by the
        # in-model moves at the end of their repeat blocks (NaN = dead leaf
        # or not recorded this iteration)
        self.d_h = np.full((self.nwalkers, self.nleaves_max), np.nan)
        self.h_h = np.full((self.nwalkers, self.nleaves_max), np.nan)
        self.in_model_proposed = np.zeros(self._counter_shape(), dtype=int)
        self.in_model_accepted = np.zeros(self._counter_shape(), dtype=int)
        self.rj_proposed = np.zeros(self._counter_shape(), dtype=int)
        self.rj_accepted = np.zeros(self._counter_shape(), dtype=int)
        self.swaps_proposed = np.zeros(self._swaps_shape(), dtype=int)
        self.swaps_accepted = np.zeros(self._swaps_shape(), dtype=int)
        self._tempered_initialized = True

    def _copy_tempered_from(self, other, dc):
        """Copy the tempered block (if any) from ``other`` using copier ``dc``."""
        if not getattr(other, "tempered_initialized", False):
            return
        for name in ("ntemps", "nwalkers", "nleaves_max", "ndim"):
            setattr(self, name, getattr(other, name))
        for name in self.tempered_array_names:
            if hasattr(other, name):
                setattr(self, name, dc(getattr(other, name)))
        self._tempered_initialized = True

    # ------------------------------------------------------------------
    # Walker-block slicing (multi-rank fan-out). Every tempered array with
    # a walker axis is sliced/merged by column. Ladders and the
    # per-iteration delta counters have NO walker axis: a slice carries the
    # ladder BY VALUE and never writes it back (the head owns ladders), and
    # its counters start at zero so a fan-out body's counts come back as
    # deltas that merge SUM-adds.
    # ------------------------------------------------------------------

    #: walker axis of every tempered array that has one (absent = no walker
    #: axis). Subclasses with a different layout override the dict.
    walker_axes: dict = {
        "coords": 1,
        "inds": 1,
        "log_like": 1,
        "log_prior": 1,
        "d_h": 0,
        "h_h": 0,
    }

    def _bare_like(self):
        """A fresh, empty instance of this class (no tempered block yet)."""
        return type(self)(None)

    def slice_walkers(self, w0: int, w1: int):
        """Copy walkers ``[w0, w1)`` of the tempered block into a fresh instance.

        Returns a bare instance when this sub-state has no tempered block.
        """
        part = self._bare_like()
        if not self.tempered_initialized:
            return part
        w0, w1 = int(w0), int(w1)
        if not (0 <= w0 < w1 <= int(self.nwalkers)):
            raise ValueError(f"walker block [{w0}, {w1}) is not inside [0, {self.nwalkers}).")
        part.initialize_tempered(
            self.ntemps,
            w1 - w0,
            self.nleaves_max,
            self.ndim,
            coords=self.coords[:, w0:w1],
            inds=self.inds[:, w0:w1],
        )
        for name, axis in self.walker_axes.items():
            if name in ("coords", "inds"):
                continue
            src = getattr(self, name, None)
            if src is None:
                continue
            getattr(part, name)[...] = np.take(src, np.arange(w0, w1), axis=axis)
        if self.betas_attr_name == "betas" and getattr(self, "betas", None) is not None:
            part.betas = np.array(self.betas, copy=True)
        return part

    def merge_walkers(self, part, w0: int, w1: int, *, sum_counters: bool = True):
        """Write ``part``'s walker columns back into ``[w0, w1)``.

        Ladders are NOT written back; delta counters are SUM-added when
        ``sum_counters`` (a slice starts from zero, so its counters are the
        deltas of the command that produced it).
        """
        part_initialized = getattr(part, "tempered_initialized", False)
        if not self.tempered_initialized and not part_initialized:
            return
        if self.tempered_initialized and not part_initialized:
            raise ValueError(
                f"merge_walkers got an uninitialized `part` for walker block [{w0}, {w1}) "
                "but `self` already has a tempered block; the update would be silently lost."
            )
        w0, w1 = int(w0), int(w1)
        if not (0 <= w0 < w1 <= int(self.nwalkers)):
            raise ValueError(f"walker block [{w0}, {w1}) is not inside [0, {self.nwalkers}).")
        if int(part.nwalkers) != w1 - w0:
            raise ValueError(
                f"slice has {part.nwalkers} walkers but the block [{w0}, {w1}) has {w1 - w0}."
            )
        for name, axis in self.walker_axes.items():
            dst = getattr(self, name, None)
            src = getattr(part, name, None)
            if dst is None or src is None:
                continue
            index = [slice(None)] * dst.ndim
            index[axis] = slice(w0, w1)
            dst[tuple(index)] = src
        if sum_counters:
            for name in self.delta_counter_names:
                dst = getattr(self, name, None)
                src = getattr(part, name, None)
                if dst is not None and src is not None:
                    dst[...] += src

    #: Per-rung STATE arrays cloned by :meth:`reseed_cold_into_hottest`.
    #: ``d_h``/``h_h`` are cold-chain-only (no temperature axis) so the
    #: hottest-rung reseed never touches them; ``betas`` defines the ladder and
    #: the ``*_proposed``/``*_accepted`` counters drive ladder adaptation --
    #: all three are deliberately left intact.
    _reseed_state_arrays: tuple = ("coords", "inds", "log_like", "log_prior")

    def reseed_cold_into_hottest(
        self, perm=None, rng=None, *, cold_rung: int = 0, hot_rung=None
    ):
        """Literally copy the cold rung's per-walker state into the hottest rung.

        For each per-rung state array ``A`` (``coords``/``inds``/``log_like``/
        ``log_prior``), ``A[hot_rung] = A[cold_rung][perm]`` -- a walker-permuted
        *copy* of the cold ensemble into the highest-temperature rung. Every
        other rung (the cold rung included) is left byte-for-byte unchanged, as
        are ``betas`` and the adaptation counters. ``d_h``/``h_h`` carry no
        temperature axis (cold-chain only), so they are never involved.

        The raw ``log_like``/``log_prior`` are temperature-independent (the
        tempered acceptance applies ``beta`` separately), so copying the cold
        rung's values into the hottest rung keeps that rung self-consistent
        with the cloned coords/inds -- no recompute is needed.

        SEARCH-ONLY: this is a deterministic injection, not a detailed-balance
        move. Callers must gate it to search stages (the standing search-waiver
        policy).

        Args:
            perm: length-``nwalkers`` permutation of the cold walkers. If
                omitted, one is drawn from ``rng`` (or a fresh ``default_rng``).
            rng: numpy ``Generator`` used to draw ``perm`` when it is ``None``.
            cold_rung: source rung index (default ``0``, the cold chain).
            hot_rung: destination rung index (default ``ntemps - 1``).
        """
        if not self.tempered_initialized:
            return
        if hot_rung is None:
            hot_rung = int(self.ntemps) - 1
        if int(hot_rung) == int(cold_rung):
            return  # single-rung ladder (or degenerate request): nothing to do
        if perm is None:
            rng = rng if rng is not None else np.random.default_rng()
            perm = rng.permutation(int(self.nwalkers))
        from lisatools.utils.utility import get_array_module

        xp = get_array_module(self.coords)
        perm = xp.asarray(perm)
        for name in self._reseed_state_arrays:
            A = getattr(self, name, None)
            if A is None:
                continue
            A[hot_rung] = A[cold_rung][perm]

    @property
    def branch(self) -> eryn_Branch:
        """An eryn ``Branch`` VIEW over this sub-state's coords/inds (shared memory)."""
        return eryn_Branch(self.coords, inds=self.inds)

    def sync_cold_row(self, main_state, branch_name: str):
        """Write this sub-state's cold row (temp 0) into the main state."""
        main_branch = main_state.branches[branch_name]
        main_branch.coords[0] = self.coords[0]
        main_branch.inds[0] = self.inds[0]

    def pull_cold_row(self, main_state, branch_name: str):
        """Write the main state's cold row (temp 0) into this sub-state's row 0.

        Inverse of :meth:`sync_cold_row` — for moves that sample the main
        (engine) state directly (e.g. the ridge-Gibbs fiber move) and must
        hand the result back to the module's tempered ladder, which the
        band moves treat as authoritative.
        """
        main_branch = main_state.branches[branch_name]
        self.coords[0] = main_branch.coords[0]
        self.inds[0] = main_branch.inds[0]

    def check_cold_row(self, main_state, branch_name: str):
        """Verify the main state's cold row matches this sub-state's row 0.

        Raises:
            ValueError: labeled description of the mismatch (inds or coords).
        """
        main_branch = main_state.branches[branch_name]
        if not np.array_equal(main_branch.inds[0], self.inds[0]):
            n_bad = int(np.sum(main_branch.inds[0] != self.inds[0]))
            raise ValueError(
                f"[{branch_name}] cold-chain inds mismatch between the main "
                f"state and its sub-state ({n_bad} differing leaf slots). "
                "A move updated one representation without the other."
            )
        main_alive = main_branch.coords[0][main_branch.inds[0]]
        sub_alive = self.coords[0][self.inds[0]]
        if not np.array_equal(main_alive, sub_alive):
            n_bad = int(np.sum(np.any(main_alive != sub_alive, axis=-1)))
            raise ValueError(
                f"[{branch_name}] cold-chain coords mismatch between the "
                f"main state and its sub-state ({n_bad} of {len(sub_alive)} "
                "alive leaves differ). A move updated one representation "
                "without the other."
            )

    def pull_from_main(self, main_state, branch_name: str):
        """Mirror the main state's full ensemble for this branch into the sub-state.

        Initializes the tempered block from the main branch on first use;
        afterwards copies coords/inds at every temperature (the Phase-2
        dual-representation sync).
        """
        main_branch = main_state.branches[branch_name]
        if not self.tempered_initialized:
            self.initialize_tempered(
                main_branch.ntemps,
                main_branch.nwalkers,
                main_branch.nleaves_max,
                main_branch.ndim,
                coords=main_branch.coords,
                inds=main_branch.inds,
            )
            return
        self.coords[:] = main_branch.coords
        self.inds[:] = main_branch.inds

    # ------------------------------------------------------------------
    # Storage contract
    # ------------------------------------------------------------------

    def tempered_storage_arrays(self) -> dict:
        """The standard tempered dict (``chain``/``inds``/logL/logP + counters)."""
        if not self.tempered_initialized:
            return {}
        out = {
            "chain": self.coords,
            "inds": self.inds,
            "log_like": self.log_like,
            "log_prior": self.log_prior,
        }
        if self.betas_attr_name == "betas" and getattr(self, "betas", None) is not None:
            out["betas"] = self.betas
        for name in ("d_h", "h_h"):
            if getattr(self, name, None) is not None:
                out[name] = getattr(self, name)
        for name in self.delta_counter_names:
            out[name] = getattr(self, name)
        return out

    def _load_tempered_from_stored(self, arrays):
        """Fill the tempered block from one stored iteration (leading axis 1)."""
        if "chain" not in arrays:
            return
        coords = np.asarray(arrays["chain"][0])
        inds = np.asarray(arrays["inds"][0]).astype(bool)
        self.initialize_tempered(*coords.shape, coords=coords, inds=inds)
        # some branches (GB) store only chain/inds -- band_info carries the
        # rest of their tempering record
        for name in (
            "log_like",
            "log_prior",
            "betas",
            "d_h",
            "h_h",
        ) + self.delta_counter_names:
            if name in arrays and getattr(self, name, None) is not None:
                getattr(self, name)[:] = arrays[name][0]

    def storage_arrays(self) -> dict:
        """``{on-disk name: array}`` persisted every saved iteration."""
        return self.tempered_storage_arrays()

    def static_arrays(self) -> dict:
        """``{on-disk name: array}`` persisted once at backend reset."""
        return {}

    def storage_attrs(self) -> dict:
        """``{name: scalar}`` written as HDF5 group attributes at reset."""
        if not self.tempered_initialized:
            return {}
        return {
            "ntemps": self.ntemps,
            "nwalkers": self.nwalkers,
            "nleaves_max": self.nleaves_max,
            "ndim": self.ndim,
        }

    @classmethod
    def make_template(cls, nwalkers, ntemps, nleaves_max=None, ndim=None, **dims):
        """Allocate a zeroed instance from dimension kwargs (extras ignored)."""
        template = cls(None)
        if _scalar_or_none(nleaves_max) is None or _scalar_or_none(ndim) is None:
            raise ValueError("Must provide nleaves_max and ndim kwargs.")
        template.initialize_tempered(ntemps, nwalkers, nleaves_max, ndim)
        return template

    @classmethod
    def from_stored(cls, arrays, statics=None, attrs=None):
        """Rebuild an instance from one stored iteration's arrays."""
        instance = cls(None)
        instance._load_tempered_from_stored(arrays)
        return instance

    def reset_delta_counters(self):
        """Zero per-iteration-delta counters (called after each save)."""
        if self.tempered_initialized:
            for name in self.delta_counter_names:
                getattr(self, name)[:] = 0

    @property
    def reset_kwargs(self):
        """Kwargs passed back to the backend when re-initializing the state."""
        out = {name: value for name, value in self.storage_attrs().items()}
        out.update(self.static_arrays())
        return out


class GBState(ModuleSubState):
    """Galactic-binary (GB) sampler state with per-band bookkeeping.

    Tracks per-band temperature ladders, swap counters, and binary-count
    arrays that the GB special moves use to drive the band-temperature
    sampler.

    Args:
        possible_state: Existing :class:`GBState` or a state-like object to
            initialize from. When it is already a :class:`GBState`, band info
            is copied over.
        band_info: Optional pre-built band-information dict.
        copy: If ``True``, deep-copy the band info from ``possible_state``.
    """

    # copy this still for each. At general hdf5 function to deal with these setups rather than specific
    @property
    def band_initialized(self):
        """Whether band tracking has been initialized for this state."""
        if hasattr(self, "band_info") and "initialized" in self.band_info:
            return self.band_info["initialized"]
        else:
            return False

    def __init__(self, possible_state, band_info=None, copy=False, **kwargs):

        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            if possible_state.band_initialized and hasattr(possible_state, "band_info"):
                self.band_info = dc(possible_state.band_info)
            self._copy_tempered_from(possible_state, dc)
        elif band_info is not None:
            self.band_info = band_info

    # ------------------------------------------------------------------
    # Per-source IDs (GB_CHOL_CACHE keys; see globalfit/moves/gb_chol_cache.py)
    #
    # ``source_uid`` (ntemps, nwalkers, nleaves_max) int64, -1 = no ID: one
    # per leaf slot, beside the per-leaf ``d_h``/``h_h``. It rides the
    # tempered block (copied with it, walker-sliced and merged with it) and
    # the GB moves keep it attached to its source (sorter build, RJ, replace,
    # write-back repack). It is per-process state and deliberately NOT in
    # ``storage_arrays``: the cache it keys is empty after a relaunch, and a
    # loaded block starts with no IDs (issued at its first sorter build).
    # ------------------------------------------------------------------
    tempered_array_names = ModuleSubState.tempered_array_names + ("source_uid",)
    walker_axes = {**ModuleSubState.walker_axes, "source_uid": 1}

    def initialize_tempered(self, ntemps, nwalkers, nleaves_max, ndim, coords=None, inds=None):
        """The base allocation plus ``source_uid`` (all ``-1``).

        A refill of an existing block (``coords`` / ``inds`` given) replaces
        the leaves wholesale, so no ID survives it.
        """
        fresh = not self.tempered_initialized
        super().initialize_tempered(ntemps, nwalkers, nleaves_max, ndim, coords=coords, inds=inds)
        if fresh or coords is not None or inds is not None:
            self.source_uid = np.full(self.inds.shape, -1, dtype=np.int64)

    def ensure_source_uid(self) -> np.ndarray:
        """The per-leaf source IDs, allocated (all ``-1``) on an instance that
        predates them (e.g. an older mid-iteration checkpoint pickle)."""
        uid = getattr(self, "source_uid", None)
        if uid is None or uid.shape != self.inds.shape:
            uid = self.source_uid = np.full(self.inds.shape, -1, dtype=np.int64)
        return uid

    def reseed_cold_into_hottest(self, perm=None, rng=None, *, cold_rung: int = 0, hot_rung=None):
        """The base reseed; the reseeded rung's sources are NEW sources, so
        they carry no ID (fresh ones are issued at the next sorter build)."""
        super().reseed_cold_into_hottest(perm=perm, rng=rng, cold_rung=cold_rung, hot_rung=hot_rung)
        if not self.tempered_initialized:
            return
        hot = int(self.ntemps) - 1 if hot_rung is None else int(hot_rung)
        if hot != int(cold_rung):
            self.ensure_source_uid()[hot] = -1

    @property
    def band_info_keys(self):
        """List of required keys for the :attr:`band_info` dict."""
        return [
            "initialized",
            "band_edges",
            "band_temps",
            "band_swaps_proposed",
            "band_swaps_accepted",
            "band_num_proposed",
            "band_num_accepted",
            "band_num_proposed_rj",
            "band_num_accepted_rj",
            "band_num_binaries",
        ]

    @property
    def band_info(self):
        """Dict holding per-band counters, temperatures, and edges."""
        return self._band_info

    @band_info.setter
    def band_info(self, band_info):
        assert isinstance(band_info, dict)
        for key in self.band_info_keys:
            if key not in band_info and key != "initialized":
                raise ValueError(f"Missing required key: {key}, for band information.")
        self._band_info = band_info
        self._band_info["initialized"] = True

    def initialize_band_information(
        self, nwalkers, ntemps, band_edges, band_temps, cap_edges=None,
        branch_name=None, leaf_caps=True, leaf_cap_per_walker=False,
        search_stage_per_walker=False, search_shutoff_per_walker=False,
    ):
        """Allocate the band-info dict with zeroed counters.

        Args:
            nwalkers: Number of MCMC walkers.
            ntemps: Number of temperatures in the ladder.
            band_edges: 1D array of frequency-band edges.
            band_temps: ``(num_bands, ntemps)`` array of inverse temperatures.
            cap_edges: Optional 1D array of LEAF-CAP cell edges, a refinement
                of ``band_edges`` (see :func:`make_cap_edges`). ``None``
                (default) means the cap grid IS the band grid, i.e. divisor 1
                and the pre-2026-08-15 behaviour.
            branch_name: Optional branch label (``"gb"`` / ``"vgb"``) used
                only to make the resume-time messages nameable.
            leaf_caps: ``False`` = this branch carries NO leaf-cap CELL state
                at all (user ruling 2026-08-22: caps gate RJ births and the
                VGB branch has no RJ surface, so it must not allocate, check,
                or persist a cap grid). Fresh stores get no ``cap_edges`` /
                ``num_cap_cells`` keys; on RESUME the cap-grid consistency
                check is skipped and any stored cap keys are dropped from the
                loaded band info, so a store whose (inert) cap datasets
                predate a band-grid migration resumes cleanly without a cap
                migration. The band-level ``band_leaf_cap`` family stays (the
                monitor reads it; it is sentinel/-1 and never consulted
                without RJ).
            leaf_cap_per_walker: ``True`` (``GBSettings.leaf_cap_per_walker``
                / ``GB_LEAF_CAP_PER_WALKER``) additionally allocates the
                ``(nwalkers, num_cells)`` cap family -- see
                :func:`ensure_cap_cell_fields`. Allocated HERE, at state
                init, rather than lazily at first gate access: the module
                backend derives its dataset set from ``storage_arrays()``
                when it resets, so an array that first appears mid-run
                would have no dataset to be written into. Default ``False``
                keeps a fresh store byte-identical to one written before
                this flag existed.
            search_stage_per_walker: ``True``
                (``GBSettings.search_stage_per_walker`` /
                ``GB_SEARCH_STAGE_PER_WALKER``) additionally allocates the
                ``(nwalkers, num_bands)`` search-stage record -- see
                :func:`ensure_search_stage_fields`. Allocated HERE for the
                same reason the per-walker cap family is: the module
                backend derives its dataset set from ``storage_arrays()``
                at reset, so an array that first appears mid-run would have
                no dataset to be written into. Default ``False`` keeps a
                fresh store byte-identical to one written before this flag
                existed.
            search_shutoff_per_walker: ``True``
                (``GBSettings.search_shutoff_per_walker`` /
                ``GB_SEARCH_BAND_SHUTOFF_PER_WALKER``) additionally
                allocates the per-(walker, band) RJ shutoff valve -- see
                :func:`ensure_search_shutoff_fields`. Independent of
                ``search_stage_per_walker``: the two features share a
                census and an update point but are separately switchable.

        Returns:
            int: the rung count that is ACTUALLY in effect after this call.
            On a fresh start that is ``ntemps``; on a RESUME it is the rung
            count carried by the stored ladder, which wins (see the resume
            branch below). Callers that size anything off the ladder
            (``TemperatureControl``, per-move ``accepted`` arrays, the
            branch's ``betas``) must use this return value rather than the
            ``ntemps`` they passed in.
        """
        if cap_edges is None:
            cap_edges = np.asarray(band_edges, dtype=float).copy()

        if not self.band_initialized:
            band_info = {}
            band_info["nwalkers"], band_info["ntemps"], band_info["band_edges"] = (
                nwalkers,
                ntemps,
                band_edges,
            )
            band_info["num_bands"] = len(band_info["band_edges"]) - 1
            if leaf_caps:
                band_info["cap_edges"] = np.asarray(cap_edges, dtype=float)
                band_info["num_cap_cells"] = len(band_info["cap_edges"]) - 1

            assert band_temps.shape == (band_info["num_bands"], band_info["ntemps"])
            band_info["band_temps"] = band_temps

            band_info["band_swaps_proposed"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"] - 1), dtype=int
            )
            band_info["band_swaps_accepted"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"] - 1), dtype=int
            )

            band_info["band_num_proposed"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"]), dtype=int
            )
            band_info["band_num_accepted"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"]), dtype=int
            )

            band_info["band_num_proposed_rj"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"]), dtype=int
            )
            band_info["band_num_accepted_rj"] = np.zeros(
                (band_info["num_bands"], band_info["ntemps"]), dtype=int
            )

            band_info["band_num_binaries"] = np.zeros(
                (band_info["ntemps"], band_info["nwalkers"], band_info["num_bands"]),
                dtype=int,
            )
            ensure_leaf_cap_fields(
                band_info, band_info["num_bands"],
                per_walker=bool(leaf_caps and leaf_cap_per_walker))
            ensure_band_shutoff_fields(band_info, band_info["num_bands"])
            if leaf_caps:
                ensure_cap_cell_fields(
                    band_info, band_info["num_cap_cells"],
                    staggered=_cap_grid_is_staggered(band_info),
                    per_walker=bool(leaf_cap_per_walker))
            # Gated on ``leaf_caps`` as well: the stage latch exists to move
            # the SEARCH's RJ-birth floors, and a branch with no RJ surface
            # (VGB) has nothing for it to gate.
            ensure_search_stage_fields(
                band_info, band_info["num_bands"],
                per_walker=bool(leaf_caps and search_stage_per_walker))
            ensure_search_shutoff_fields(
                band_info, band_info["num_bands"],
                per_walker=bool(leaf_caps and search_shutoff_per_walker))
            band_info["initialized"] = True
            self.band_info = band_info
            return int(band_info["ntemps"])

        else:
            # already initialized: validate the geometry is unchanged.
            # band_info dicts that round-tripped through the HDF backend
            # (GBHDFBackend stores only the ``band_info_keys`` arrays)
            # lack the nwalkers/ntemps/num_bands scalars -- backfill them
            # from the array shapes before validating.
            bi = self.band_info
            # Arrays loaded through GBHDFBackend.get_band_info keep the
            # backend's leading step axis; in-run consumers index the bare
            # per-iteration shapes, so strip it. Rank-based (not
            # shape[0]==1) so genuine single-band/single-temp axes survive.
            _bare_ndim = {
                "band_temps": 2, "band_num_proposed": 2, "band_num_accepted": 2,
                "band_num_proposed_rj": 2, "band_num_accepted_rj": 2,
                "band_swaps_proposed": 2, "band_swaps_accepted": 2,
                "band_num_binaries": 3, "band_leaf_cap": 1,
                "band_cap_iters": 1, "band_best_ll": 1,
                # the RJ shutoff valve's persisted record (2026-08-29)
                "band_occ_streak": 1, "band_occ_last": 1,
                "band_rj_shutoff": 1, "band_shutoff_since_revive": 1,
                "band_shutoff_epoch": 1,
                "band_cold_ll": 2,
                "cap_cell_leaf_cap": 1, "cap_cell_iters": 1,
                "cap_cell_best_ll": 1, "cap_cell_cold_ll": 2,
                # the per-walker cap family (GB_LEAF_CAP_PER_WALKER)
                "cap_cell_leaf_cap_w": 2, "cap_cell_iters_w": 2,
                "cap_cell_best_ll_w": 2, "band_best_ll_w": 2,
                # the per-(walker, band) search-stage record
                # (GB_SEARCH_STAGE_PER_WALKER) + its min-over-walkers mirror
                "band_stage_w": 2, "band_stage_occ_last_w": 2,
                "band_stage_streak_w": 2, "band_stage": 1,
                # the per-(walker, band) RJ shutoff valve + its window
                "band_rj_shutoff_w": 2, "band_shutoff_w_step": 1,
                "band_shutoff_best_w": 2, "band_shutoff_streak_w": 2,
                "band_cold_logl_max_w": 2, "band_cold_logl_w": 2,
                "band_shutoff_reset_w": 2, "band_cold_logl_peak_w": 2,
                # the valve's window-rule stamp (2026-10-13)
                VALVE_WINDOW_RULE_FIELD: 1,
            }
            for _key, _nd in _bare_ndim.items():
                _arr = bi.get(_key)
                if isinstance(_arr, np.ndarray) and _arr.ndim == _nd + 1:
                    bi[_key] = _arr[-1]
            bi.setdefault("num_bands", len(bi["band_edges"]) - 1)
            # Stores written before the cap grid existed carry no
            # ``cap_edges``: their cap grid IS the band grid (divisor 1),
            # which is exactly what the guard below then compares against.
            bi.setdefault("cap_edges", np.asarray(bi["band_edges"], dtype=float).copy())
            bi["num_cap_cells"] = len(bi["cap_edges"]) - 1
            bi.setdefault("ntemps", int(bi["band_temps"].shape[-1]))
            bi.setdefault("nwalkers", int(bi["band_num_binaries"].shape[-2]))
            # RESUME. ``per_walker`` here is the ENABLE path as well as the
            # restore path: a store written without the flag has no ``_w``
            # arrays, so turning the flag on seeds every walker by broadcast
            # from the stored shared cap (see ensure_cap_cell_fields). A
            # store that HAS them keeps them -- setdefault never overwrites.
            ensure_leaf_cap_fields(
                bi, bi["num_bands"],
                per_walker=bool(leaf_caps and leaf_cap_per_walker))
            ensure_cap_cell_fields(bi, bi["num_cap_cells"],
                                   staggered=_cap_grid_is_staggered(bi),
                                   per_walker=bool(leaf_cap_per_walker))
            # RESUME path for the shutoff valve: a stored record whose grid
            # no longer matches is DISCARDED here rather than restored onto
            # a grid it was never measured on (see the function's docstring
            # for the full degradation table).
            ensure_band_shutoff_fields(bi, bi["num_bands"])
            # RESUME path for the stage latch. Unlike the cap family this is
            # NOT an enable-by-broadcast seam: there is no shared 1-D stage
            # to widen, so turning the flag on over an existing store starts
            # every (walker, band) at COARSE. That is the correct direction
            # -- a store that never measured the latch cannot hand a walker
            # the relaxed floor it did not earn.
            _stage_origin = ensure_search_stage_fields(
                bi, bi["num_bands"],
                per_walker=bool(leaf_caps and search_stage_per_walker))
            if _stage_origin not in ("off", "fresh"):
                logger.info("[GB_STAGE] search-stage record %s", _stage_origin)
            _shutoff_w_origin = ensure_search_shutoff_fields(
                bi, bi["num_bands"],
                per_walker=bool(leaf_caps and search_shutoff_per_walker))
            if _shutoff_w_origin not in ("off", "fresh"):
                logger.info("[GB_STAGE] per-walker RJ shutoff valve %s",
                            _shutoff_w_origin)
            _label = f"branch {branch_name!r}" if branch_name else "banded branch"
            _stored_nt = int(bi["ntemps"])
            _stored_nw = int(bi["nwalkers"])
            _stored_nb = int(bi["num_bands"])

            # ---- 1. is the STORED band_info self-consistent? ------------
            # Every rung-dimensioned array must agree with the rung count
            # the ladder itself declares, and every band-dimensioned array
            # with the stored band grid. A disagreement here is genuine
            # corruption (a half-migrated store, an interrupted re-rung),
            # not a config change -- refuse with a message that names the
            # offending array, never a bare assert.
            _expected_shapes = {
                "band_temps": (_stored_nb, _stored_nt),
                "band_num_proposed": (_stored_nb, _stored_nt),
                "band_num_accepted": (_stored_nb, _stored_nt),
                "band_num_proposed_rj": (_stored_nb, _stored_nt),
                "band_num_accepted_rj": (_stored_nb, _stored_nt),
                "band_swaps_proposed": (_stored_nb, max(_stored_nt - 1, 0)),
                "band_swaps_accepted": (_stored_nb, max(_stored_nt - 1, 0)),
                "band_num_binaries": (_stored_nt, _stored_nw, _stored_nb),
                # the per-walker cap family (GB_LEAF_CAP_PER_WALKER). A
                # stored ``_w`` array whose walker axis disagrees with the
                # ladder would gate walker w against some other walker's
                # allowance -- silently, since every index stays in range.
                "cap_cell_leaf_cap_w": (
                    _stored_nw, int(bi.get("num_cap_cells", 0))),
                "cap_cell_iters_w": (
                    _stored_nw, int(bi.get("num_cap_cells", 0))),
                "cap_cell_best_ll_w": (
                    _stored_nw, int(bi.get("num_cap_cells", 0))),
                "band_best_ll_w": (_stored_nw, _stored_nb),
                # the per-(walker, band) search-stage record. When the flag
                # is ON ``ensure_search_stage_fields`` has already reset any
                # mismatch to COARSE, so these never fire; they are here for
                # the flag-OFF case, where the arrays are inert but a store
                # carrying arrays that disagree with its own ladder is
                # corruption by this check's own definition and must say so
                # rather than be persisted onward. Same rule, same place, as
                # the per-walker cap family above.
                "band_stage_w": (_stored_nw, _stored_nb),
                "band_stage_occ_last_w": (_stored_nw, _stored_nb),
                "band_stage_streak_w": (_stored_nw, _stored_nb),
                "band_rj_shutoff_w": (_stored_nw, _stored_nb),
                "band_shutoff_best_w": (_stored_nw, _stored_nb),
                "band_shutoff_streak_w": (_stored_nw, _stored_nb),
                "band_cold_logl_max_w": (_stored_nw, _stored_nb),
                "band_cold_logl_w": (_stored_nw, _stored_nb),
                "band_shutoff_reset_w": (_stored_nw, _stored_nb),
                "band_cold_logl_peak_w": (_stored_nw, _stored_nb),
            }
            for _key, _want in _expected_shapes.items():
                _arr = bi.get(_key)
                if _arr is None:
                    continue
                _got = tuple(np.shape(_arr))
                if _got != _want:
                    raise ValueError(
                        f"corrupted band information for {_label}: stored "
                        f"{_key!r} has shape {_got} but the stored ladder "
                        f"declares ntemps={_stored_nt}, nwalkers="
                        f"{_stored_nw}, num_bands={_stored_nb}, i.e. shape "
                        f"{_want}. The store's own per-band arrays disagree "
                        f"with each other (half-written / half-migrated "
                        f"file). Restore the .bak written by scripts/"
                        f"fstat_proposal/fix_vgb_band_temps.py or "
                        f"migrate_gb_band_edges.py, or start a fresh "
                        f"backend -- do NOT resume this file."
                    )

            # ---- 2. walker count: a config change we cannot reconcile ----
            if int(nwalkers) != _stored_nw:
                raise ValueError(
                    f"walker-count mismatch for {_label}: the state stores "
                    f"nwalkers={_stored_nw} but the run config builds "
                    f"nwalkers={int(nwalkers)}. The per-band arrays are "
                    f"sized by the walker count, so the resume cannot be "
                    f"reconciled: restore the original nwalkers (NWALKERS) "
                    f"or start a fresh backend."
                )

            # ---- 3. rung count: the STORED ladder WINS on resume --------
            # (2026-08-15) A resumed store carries its own rung count in
            # band_temps; the branch's configured ladder (e.g. VGB_NTEMPS,
            # default 12) may differ -- typically because the store was
            # written by an older/buggier config. Re-rungging live per-band
            # temperatures + counters is a MIGRATION, not something to do
            # silently mid-resume, so the stored ladder is authoritative and
            # the configured one is reported. The return value carries the
            # resolution to the caller, which must size the move's
            # betas/temperature machinery off it.
            if int(ntemps) != _stored_nt:
                logger.warning(
                    "band-temperature ladder mismatch on resume for %s: the "
                    "state stores %d rung(s) but the run config builds %d "
                    "(GB_NTEMPS / VGB_NTEMPS or an explicit per-branch "
                    "`betas`). THE STORED %d-RUNG LADDER WINS -- this run "
                    "continues at ntemps=%d and the configured %d-rung "
                    "ladder is ignored. To actually run at %d rungs, re-rung "
                    "the store first with "
                    "scripts/fstat_proposal/fix_vgb_band_temps.py "
                    "<store.h5> %d (it recreates every rung-dimensioned "
                    "dataset and writes a .bak), or start a fresh backend.",
                    _label, _stored_nt, int(ntemps), _stored_nt, _stored_nt,
                    int(ntemps), int(ntemps), int(ntemps),
                )
            # Band-grid geometry check. Explicit (the old bare
            # ``assert np.all(==)`` compared unequal-length arrays to a
            # scalar False with a DeprecationWarning) and tolerant to float
            # round-off (a migrated store's edges may differ by ~1 ulp from
            # the settings-derived ones when layer_df was reconstructed).
            _cfg_edges = np.asarray(band_edges, dtype=float)
            _stored_edges = np.asarray(bi["band_edges"], dtype=float)
            if _cfg_edges.shape != _stored_edges.shape or not np.allclose(
                _cfg_edges, _stored_edges, rtol=1e-9, atol=0.0
            ):
                raise ValueError(
                    f"band grid mismatch: state stores "
                    f"{len(_stored_edges) - 1} sub-bands but the run config "
                    f"builds {len(_cfg_edges) - 1}. The band-edge knobs "
                    f"(GB_BAND_EDGES_MODE / GB_BAND_TARGET_COUNT / "
                    f"GB_BAND_MIN_LAYERS / GB_SUBBAND_DIVISOR) changed, or "
                    f"the store needs scripts/fstat_proposal/"
                    f"migrate_gb_band_edges.py."
                )
            # Cap-free branch (leaf_caps=False, e.g. VGB): no cap grid to
            # check, and any cap keys a stored band_info carries (written
            # before the branch went cap-free, or stranded by a band-grid
            # migration that reshaped the bands but not the cap cells) are
            # DROPPED here so the backend stops persisting them and no
            # consumer can read a stale grid.
            if not leaf_caps:
                for _cap_key in (
                    "cap_edges", "num_cap_cells", "cap_cell_leaf_cap",
                    "cap_cell_iters", "cap_cell_best_ll", "cap_cell_cold_ll",
                    "cap_cell_leaf_cap_w", "cap_cell_iters_w",
                    "cap_cell_best_ll_w",
                    # the search-stage record follows the cap family: it
                    # gates RJ-birth floors and a cap-free branch has no RJ
                    # surface for it to gate, so a stored record here is
                    # stranded state, not something to keep persisting.
                    "band_stage_w", "band_stage_occ_last_w",
                    "band_stage_streak_w", "band_stage",
                    "band_rj_shutoff_w", "band_shutoff_w_step",
                    "band_shutoff_best_w", "band_shutoff_streak_w",
                    "band_cold_logl_max_w", "band_cold_logl_w",
                    "band_shutoff_reset_w", "band_cold_logl_peak_w",
                    VALVE_WINDOW_RULE_FIELD,
                ):
                    bi.pop(_cap_key, None)
                return _stored_nt

            # Leaf-CAP grid check (user design 2026-08-15). The cap grid is
            # a refinement of the band grid by GB_CAP_DIVISOR; a store whose
            # cap grid disagrees with the configured divisor must refuse
            # loudly rather than silently resume a run whose per-cell cap
            # state means something else.
            _cfg_cap = np.asarray(cap_edges, dtype=float)
            _stored_cap = np.asarray(bi["cap_edges"], dtype=float)
            if _cfg_cap.shape != _stored_cap.shape or not np.allclose(
                _cfg_cap, _stored_cap, rtol=1e-9, atol=0.0
            ):
                _k_cfg = (len(_cfg_cap) - 1) / max(len(_cfg_edges) - 1, 1)
                _k_store = (len(_stored_cap) - 1) / max(len(_stored_edges) - 1, 1)
                raise ValueError(
                    f"leaf-cap grid mismatch: the state stores "
                    f"{len(_stored_cap) - 1} cap cells (divisor ~{_k_store:g}) "
                    f"but the run config builds {len(_cfg_cap) - 1} "
                    f"(GB_CAP_DIVISOR ~{_k_cfg:g}). NOTE: identical cell "
                    f"counts with differing edge VALUES means the STAGGER "
                    f"flag flipped (GB_CAP_STAGGER / GBSettings.cap_stagger "
                    f"-- fresh store only). Either restore the old "
                    f"GB_CAP_DIVISOR / GBSettings.cap_divisor, or migrate the "
                    f"store with scripts/fstat_proposal/migrate_gb_cap_grid.py "
                    f"(which splits each band's stored cap state into its "
                    f"children, inheriting cap + min-iters counters)."
                )

            return _stored_nt

    def update_band_information(
        self,
        band_temps,
        band_num_proposed,
        band_num_accepted,
        band_swaps_proposed,
        band_swaps_accepted,
        band_num_binaries,
        is_rj,
    ):
        """Accumulate one iteration's worth of band counters.

        Args:
            band_temps: New ``(num_bands, ntemps)`` temperature ladder.
            band_num_proposed: ``(num_bands, ntemps)`` proposal counts.
            band_num_accepted: ``(num_bands, ntemps)`` acceptance counts.
            band_swaps_proposed: ``(num_bands, ntemps - 1)`` swap proposals.
            band_swaps_accepted: ``(num_bands, ntemps - 1)`` swap acceptances.
            band_num_binaries: ``(ntemps, nwalkers, num_bands)`` binary count.
            is_rj: ``True`` to credit reversible-jump counters, otherwise
                in-model counters.
        """
        self.band_info["band_temps"][:] = band_temps
        self.band_info["band_num_binaries"][:] = band_num_binaries

        if not is_rj:
            self.band_info["band_num_proposed"] += band_num_proposed
            self.band_info["band_num_accepted"] += band_num_accepted
        else:
            self.band_info["band_num_proposed_rj"] += band_num_proposed
            self.band_info["band_num_accepted_rj"] += band_num_accepted

        self.band_info["band_swaps_proposed"] += band_swaps_proposed
        self.band_info["band_swaps_accepted"] += band_swaps_accepted

    def _accumulate_into(self, key, value) -> None:
        """``band_info[key] += value``, tolerating a host/device mismatch.

        Both callers on the multi-rank fan-out path pool their counts on
        the HOST -- the ranks ship them over MPI, so the head accumulates
        into ``np.zeros(...)`` -- while ``band_info`` may be device
        resident. CuPy refuses implicit host operands
        (``TypeError: Unsupported type <class 'numpy.ndarray'>``), which is
        what killed the ladder adaptation in job 627's rj_replace. These
        three accumulators sit a few lines PAST that crash point on the
        same propose, so they had never executed either. Coerce to the
        destination's module rather than make every caller remember.
        """
        from lisatools.utils.utility import get_array_module

        dest = self.band_info[key]
        self.band_info[key] = dest + get_array_module(dest).asarray(value)

    def accumulate_proposals(self, proposed, accepted, is_rj: bool) -> None:
        """Accumulate ``(num_bands, ntemps)`` proposal/acceptance counts into
        the RJ or in-model counter family."""
        if is_rj:
            self._accumulate_into("band_num_proposed_rj", proposed)
            self._accumulate_into("band_num_accepted_rj", accepted)
        else:
            self._accumulate_into("band_num_proposed", proposed)
            self._accumulate_into("band_num_accepted", accepted)

    def accumulate_swaps(self, proposed, accepted) -> None:
        """Accumulate ``(num_bands, ntemps - 1)`` tempering swap counts."""
        self._accumulate_into("band_swaps_proposed", proposed)
        self._accumulate_into("band_swaps_accepted", accepted)

    def reset_band_counters(self):
        """Zero all per-band proposal/acceptance/swap counters."""
        self.band_info["band_num_proposed"][:] = 0
        self.band_info["band_num_accepted"][:] = 0
        self.band_info["band_num_proposed_rj"][:] = 0
        self.band_info["band_num_accepted_rj"][:] = 0
        self.band_info["band_swaps_proposed"][:] = 0
        self.band_info["band_swaps_accepted"][:] = 0

    # ------------------------------------------------------------------
    # ModuleSubState storage contract
    # ------------------------------------------------------------------

    static_names = ("band_edges", "cap_edges")
    dim_attr_names = (
        "num_bands", "num_cap_cells", "ntemps", "nwalkers", "nleaves_max", "ndim",
    )
    # GB's ladder is per band (band_info["band_temps"]); no flat betas
    betas_attr_name = "band_temps"
    #: all band arrays keep the backend float dtype (pre-rework layout)
    legacy_dtype_names = (
        "band_edges",
        "band_temps",
        "band_swaps_proposed",
        "band_swaps_accepted",
        "band_num_proposed",
        "band_num_accepted",
        "band_num_proposed_rj",
        "band_num_accepted_rj",
        "band_num_binaries",
        "band_leaf_cap",
        "band_cap_iters",
        "band_best_ll",
        "band_cold_ll",
        "cap_edges",
        "cap_cell_leaf_cap",
        "cap_cell_iters",
        "cap_cell_best_ll",
        "cap_cell_cold_ll",
        # the per-walker cap family rides the same channel as its 1-D
        # twins, so it keeps the same backend float dtype convention
        "cap_cell_leaf_cap_w",
        "cap_cell_iters_w",
        "cap_cell_best_ll_w",
        "band_best_ll_w",
    )

    #: band_info entries that are STATIC (written once at backend reset),
    #: not per-iteration storage arrays
    _static_band_info_names = ("band_edges", "cap_edges")

    #: What a stored row written BEFORE an array existed means, where that
    #: differs from the array's fresh-state value. Read by
    #: ``ModuleSubBackend._create_missing_datasets`` for the earlier rows of
    #: a dataset it creates on a store that predates it. The valve's window
    #: rule: a fresh valve is stamped with the current rule (nothing learned),
    #: but every row written before the stamp existed was learned under the
    #: CENTRES rule -- a resume or a rewind onto such a row must re-learn.
    predating_fill_values = {VALVE_WINDOW_RULE_FIELD: VALVE_WINDOW_RULE_CENTRES}

    def storage_arrays(self):
        """Every per-band array plus the tempered ``chain``/``inds``.

        The per-branch ``log_like``/``log_prior`` and base counters are
        omitted -- ``band_info`` carries the GB tempering record
        (``band_temps`` + ``band_num_*``) at per-band resolution.
        """
        out = {
            name: dat
            for name, dat in self.band_info.items()
            if isinstance(dat, np.ndarray)
            and name not in self._static_band_info_names
        }
        if self.tempered_initialized:
            out["chain"] = self.coords
            out["inds"] = self.inds
            for name in ("d_h", "h_h"):
                if getattr(self, name, None) is not None:
                    out[name] = getattr(self, name)
        return out

    def static_arrays(self):
        return {
            "band_edges": self.band_info["band_edges"],
            "cap_edges": self.band_info.get(
                "cap_edges", self.band_info["band_edges"]
            ),
        }

    def storage_attrs(self):
        out = dict(super().storage_attrs())
        out["num_bands"] = len(self.band_info["band_edges"]) - 1
        out["num_cap_cells"] = len(self.static_arrays()["cap_edges"]) - 1
        return out

    #: The per-walker band families, keyed by the
    #: :meth:`initialize_band_information` flag that allocates each one. A
    #: flag reads ON in :attr:`reset_kwargs` when ANY of its arrays is in the
    #: live band info, so the backend's ``reset`` builds its template -- and
    #: with it the store's dataset set -- with exactly the families the live
    #: state will save. (2026-10-10, 9mo jobs 751-753: ``make_template`` ran
    #: with every flag at its default False, the store never got these
    #: datasets, ``save_step`` skipped them, and every resume restarted the
    #: per-(walker, band) RJ shutoff valve from scratch.)
    _per_walker_family_markers = {
        "leaf_cap_per_walker": ("band_best_ll_w",)
        + tuple(name for name, _ in CAP_CELL_PER_WALKER_FIELDS),
        "search_stage_per_walker": SEARCH_STAGE_FIELDS,
        "search_shutoff_per_walker": SEARCH_SHUTOFF_FIELDS,
    }

    @property
    def reset_kwargs(self):
        """The base reset kwargs plus the three per-walker family flags.

        The flags are read off the live band info rather than the settings:
        what the backend must create is what :meth:`storage_arrays` will
        hand it, whichever path allocated it.
        """
        out = dict(super().reset_kwargs)
        bi = self.band_info if self.band_initialized else {}
        for flag, names in self._per_walker_family_markers.items():
            out[flag] = any(bi.get(name) is not None for name in names)
        return out

    @classmethod
    def make_template(
        cls,
        nwalkers,
        ntemps,
        num_bands=None,
        band_edges=None,
        cap_edges=None,
        num_cap_cells=None,
        nleaves_max=None,
        ndim=None,
        branch_name=None,
        leaf_caps=True,
        leaf_cap_per_walker=False,
        search_stage_per_walker=False,
        search_shutoff_per_walker=False,
        **kwargs,
    ):
        """Allocate a zeroed GB sub-state from the backend's reset kwargs.

        ``branch_name``, ``leaf_caps`` and the three per-walker flags are
        forwarded to :meth:`initialize_band_information`, so a store created
        with a per-walker knob on gets that family's datasets from the start.
        All default to the flag-off layout (a store that never had them).
        """
        if num_bands is None or band_edges is None:
            raise ValueError("Must provide num_bands and band_edges kwargs.")
        template = cls(None)
        template.initialize_band_information(
            nwalkers, ntemps, band_edges, np.zeros((num_bands, ntemps)),
            cap_edges=cap_edges,
            branch_name=branch_name,
            leaf_caps=bool(leaf_caps),
            leaf_cap_per_walker=bool(leaf_cap_per_walker),
            search_stage_per_walker=bool(search_stage_per_walker),
            search_shutoff_per_walker=bool(search_shutoff_per_walker),
        )
        if _scalar_or_none(nleaves_max) is not None and _scalar_or_none(ndim) is not None:
            template.initialize_tempered(ntemps, nwalkers, nleaves_max, ndim)
        return template

    @classmethod
    def from_stored(cls, arrays, statics=None, attrs=None):
        # The stored band arrays keep their leading step axis; GBState's
        # initialize_band_information strips it rank-based on reload.
        band_info = {
            name: value
            for name, value in arrays.items()
            if name.startswith("band_") or name.startswith("cap_cell_")
        }
        band_info["band_edges"] = statics["band_edges"]
        # Files written before the cap grid existed have no ``cap_edges``
        # static: divisor 1, cap grid == band grid.
        band_info["cap_edges"] = statics.get("cap_edges", statics["band_edges"])
        band_info["initialized"] = True
        instance = cls(None, band_info=band_info)
        instance._load_tempered_from_stored(arrays)
        return instance

    def reset_delta_counters(self):
        self.reset_band_counters()


class PerLeafLadderState(ModuleSubState):
    """Shared base for sub-states carrying one temperature ladder per leaf.

    ``betas_all`` has shape ``(nleaves_max, ntemps)`` -- one independent
    ladder per source. Concrete classes set ``branch_name`` and the legacy
    per-branch leaf-count attribute name (``num_mbhs`` / ``num_emris`` /
    ``num_sobbhs``) via ``leaf_count_name``.

    Args:
        possible_state: Existing instance of the same class or coords-like
            dict.
        betas_all: Optional ``(nleaves_max, ntemps)`` array of inverse
            temperatures, one row per leaf.
        copy: If ``True``, deep-copy data from ``possible_state``.
    """

    branch_name: str = None
    leaf_count_name: str = None
    remove_kwargs = ["betas_all"]

    def __init__(self, possible_state, betas_all=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            self.betas_all = dc(possible_state.betas_all)
            self._set_leaf_count(getattr(possible_state, self.leaf_count_name))
            self._copy_tempered_from(possible_state, dc)
        else:
            self.betas_all = betas_all
            if possible_state is None:
                # HDF warm-start: from_stored passes possible_state=None and
                # only betas_all (nleaves, ntemps) — its second-to-last axis
                # is the leaf count. The coords live in the main GFState, so
                # there is no branch to index here.
                self._set_leaf_count(
                    betas_all.shape[-2] if betas_all is not None else 20
                )
            else:
                self._set_leaf_count(
                    branch_nleaves_max(possible_state, self.branch_name)
                )

    def _set_leaf_count(self, n):
        setattr(self, self.leaf_count_name, int(n))

    @property
    def num_leaves(self):
        """Leaf count under its generic name (aliases ``num_mbhs`` etc.)."""
        return getattr(self, self.leaf_count_name)

    # ------------------------------------------------------------------
    # ModuleSubState storage contract
    # ------------------------------------------------------------------

    legacy_dtype_names = ("betas_all",)
    # the ladder is per leaf (betas_all); no flat betas
    betas_attr_name = "betas_all"

    # per-leaf log_like / log_prior are (nleaves_max, ntemps, nwalkers): walker axis LAST
    walker_axes = {**ModuleSubState.walker_axes, "log_like": 2, "log_prior": 2}

    def _bare_like(self):
        betas_all = None if self.betas_all is None else np.array(self.betas_all, copy=True)
        return type(self)(None, betas_all=betas_all)

    # per-leaf resolution: each leaf carries its own ladder, likelihood
    # rows, and counters
    def _ll_shape(self):
        return (self.nleaves_max, self.ntemps, self.nwalkers)

    def _counter_shape(self):
        return (self.nleaves_max, self.ntemps)

    def _swaps_shape(self):
        return (self.nleaves_max, max(self.ntemps - 1, 0))

    def storage_arrays(self):
        out = {"betas_all": self.betas_all}
        out.update(self.tempered_storage_arrays())
        return out

    def storage_attrs(self):
        out = dict(super().storage_attrs())
        out[self.leaf_count_name] = self.num_leaves
        return out

    @classmethod
    def make_template(cls, nwalkers, ntemps, nleaves_max=None, ndim=None, **dims):
        num_leaves = _scalar_or_none(dims.get(cls.leaf_count_name))
        if num_leaves is None:
            num_leaves = _scalar_or_none(nleaves_max)
        if num_leaves is None:
            raise ValueError(f"Must provide {cls.leaf_count_name} kwarg.")
        template = cls(None, betas_all=np.zeros((num_leaves, ntemps)))
        if _scalar_or_none(ndim) is not None:
            template.initialize_tempered(ntemps, nwalkers, num_leaves, ndim)
        return template

    @classmethod
    def from_stored(cls, arrays, statics=None, attrs=None):
        # [0] squeezes the (single) iteration axis to the live
        # (nleaves, ntemps) shape.
        instance = cls(None, betas_all=arrays["betas_all"][0])
        instance._load_tempered_from_stored(arrays)
        return instance


class MBHState(PerLeafLadderState):
    """Massive black-hole binary sampler state with per-leaf temperature ladder."""

    branch_name = "mbh"
    leaf_count_name = "num_mbhs"
    dim_attr_names = ("num_mbhs", "ntemps", "nwalkers", "nleaves_max", "ndim")


class EMRIState(PerLeafLadderState):
    """Extreme mass-ratio inspiral sampler state with per-leaf temperature ladder."""

    branch_name = "emri"
    leaf_count_name = "num_emris"
    dim_attr_names = ("num_emris", "ntemps", "nwalkers", "nleaves_max", "ndim")


class SOBBHState(PerLeafLadderState):
    """Stellar-origin BBH (SOBBH) sampler state with per-leaf temperature ladder.

    Mirrors :class:`EMRIState` — one row of ``betas_all`` per SOBBH leaf so each
    source carries its own tempering ladder.
    """

    branch_name = "sobbh"
    leaf_count_name = "num_sobbhs"
    dim_attr_names = ("num_sobbhs", "ntemps", "nwalkers", "nleaves_max", "ndim")


class GFState(eryn_State):
    """Composite global-fit state holding per-source-class sub-states.

    Wraps an :class:`eryn.state.State` with a dict mapping each branch name
    (``gb``, ``mbh``, ``emri``, ...) to an associated state subclass
    (e.g. :class:`GBState`, :class:`MBHState`).

    Args:
        possible_state: Either an existing :class:`GFState` to copy from or a
            coords-like input.
        is_eryn_state_input: When ``True``, treat ``possible_state`` as a
            plain :class:`eryn.state.State` rather than a :class:`GFState`.
        sub_state_bases: Mapping ``{branch_name: state_class}`` giving the
            sub-state class to instantiate for each branch.
    """

    # TODO: bandaid fix this
    def __init__(
        self,
        possible_state,
        *args,
        is_eryn_state_input: bool = False,
        sub_state_bases: dict = None,
        **kwargs,
    ):

        eryn_State.__init__(self, possible_state, *args, **kwargs)
        self.sub_states = {}
        if isinstance(possible_state, type(self)) and not is_eryn_state_input:
            self.sub_state_bases = possible_state.sub_state_bases
            for name in self.branches:
                sub_state_base = self.sub_state_bases.get(name, None)
                if sub_state_base is not None:
                    self.sub_states[name] = sub_state_base(
                        possible_state.sub_states[name], *args, **kwargs
                    )
                else:
                    self.sub_states[name] = None

        else:
            self.sub_state_bases = sub_state_bases

            for name in self.branches:
                if sub_state_bases is not None and sub_state_bases[name] is not None:
                    self.sub_states[name] = sub_state_bases[name](
                        possible_state,  # this is just coords in the first input
                        *args,
                        **kwargs,
                    )
                else:
                    self.sub_states[name] = None

        # elif sub_state_bases is None and is_eryn_state_input:
        #     raise ValueError

        # elif is_eryn_state_input:
        #     self.sub_state_bases = sub_state_bases
        #     for name in self.branches:
        #         sub_state_base = sub_state_bases.get(name, None)
        #         if sub_state_base is not None:
        #             self.sub_states[name] = sub_state_base(
        #                 None,
        #                 *args,
        #                 **kwargs
        #             )
        #         else:
        #             self.sub_states[name] = None


class AllGFBranchInfo:
    """Aggregate of two or more :class:`GFBranchInfo` instances.

    Combines per-branch metadata dicts (``ndims``, ``nleaves_max``,
    ``nleaves_min``, ``branch_state``, ``branch_backend``) so that a global
    fit can query branch info uniformly regardless of how many sources are
    in the model. Use the ``+`` operator on :class:`GFBranchInfo` /
    :class:`AllGFBranchInfo` to chain them together.
    """

    def __init__(self, branch_1, branch_2):

        for key in [
            "name",
            "ndims",
            "nleaves_max",
            "nleaves_min",
            "branch_state",
            "branch_backend",
        ]:
            if isinstance(branch_1, AllGFBranchInfo) and isinstance(branch_2, AllGFBranchInfo):
                if key == "name":
                    self.branch_names = branch_1.branch_names + branch_2.name
                    continue
                setattr(self, key, {**getattr(branch_1, key), **getattr(branch_2, key)})

            elif isinstance(branch_1, GFBranchInfo) and isinstance(branch_2, GFBranchInfo):
                if key == "name":
                    self.branch_names = [branch_1.name, branch_2.name]
                    continue
                setattr(
                    self,
                    key,
                    {
                        branch_1.name: getattr(branch_1, key),
                        branch_2.name: getattr(branch_2, key),
                    },
                )
            else:
                if not isinstance(branch_2, GFBranchInfo):
                    # switch so all branch is in position 1
                    tmp = branch_1
                    branch_1 = branch_2
                    branch_2 = tmp
                if key == "name":
                    self.branch_names = branch_1.branch_names + [branch_2.name]
                    continue
                setattr(
                    self,
                    key,
                    {**getattr(branch_1, key), branch_2.name: getattr(branch_2, key)},
                )

    def __add__(self, branch_2):
        return AllGFBranchInfo(self, branch_2)

    @property
    def ndims(self):
        return self._ndims

    @ndims.setter
    def ndims(self, ndims):
        assert isinstance(ndims, dict)
        self._ndims = ndims

    @property
    def branch_names(self):
        return self._branch_names

    @branch_names.setter
    def branch_names(self, branch_names):
        assert isinstance(branch_names, list)
        self._branch_names = branch_names

    @property
    def nleaves_max(self):
        return self._nleaves_max

    @nleaves_max.setter
    def nleaves_max(self, nleaves_max):
        assert isinstance(nleaves_max, dict)
        self._nleaves_max = nleaves_max

    @property
    def nleaves_min(self):
        return self._nleaves_min

    @nleaves_min.setter
    def nleaves_min(self, nleaves_min):
        assert isinstance(nleaves_min, dict)
        self._nleaves_min = nleaves_min

    @property
    def branch_state(self):
        return self._branch_state

    @branch_state.setter
    def branch_state(self, branch_state):
        self._branch_state = branch_state

    @property
    def branch_backend(self):
        return self._branch_backend

    @branch_backend.setter
    def branch_backend(self, branch_backend):
        self._branch_backend = branch_backend


from eryn.backends import backend as eryn_Backend


@dataclass
class GFBranchInfo:
    """Metadata describing a single branch in the global fit.

    Args:
        name: Branch name (e.g. ``"gb"``, ``"mbh"``).
        ndims: Number of parameters per leaf.
        nleaves_max: Maximum allowed leaves on this branch.
        nleaves_min: Minimum allowed leaves on this branch.
        branch_state: Optional state class associated with this branch.
        branch_backend: Optional backend object associated with this branch.
    """

    name: str
    ndims: int
    nleaves_max: int
    nleaves_min: int
    branch_state: eryn_State = None
    branch_backend: eryn_Backend = None

    def __add__(self, branch_2):
        return AllGFBranchInfo(self, branch_2)
