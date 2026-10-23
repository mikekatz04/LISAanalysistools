"""Host-side cache of GB in-model proposal factors (``GB_CHOL_CACHE``).

Where it sits: every GB in-model block of
:class:`~lisatools.globalfit.moves.gbspecialstretch.GBSpecialBase` asks
``_proposal_cholesky`` for its sources' proposal factors. When the shared
nearest-f0 table is not in use (``GB_INFOMAT_PER_BLOCK=1``, the per-block exact
route) and the move's ``_chol_cache`` gate passes (``GB_CHOL_CACHE=1``, an
info-matrix proposal, a GB branch without per-leaf fills -- not VGB), the
factors come from :meth:`_CholCache.take` instead of a direct
``_compute_proposal_cholesky`` call. ``_ensure_proposal_tables`` runs
:meth:`_CholCache.due` / :meth:`_CholCache.refresh` once per proposal, and
``_run_in_model_repeats`` calls :meth:`_CholCache.retrack` at the end of each
block. The 9mo/12mo launchers set ``GB_CHOL_CACHE=1``,
``GB_CHOL_CACHE_EVERY=40``, ``GB_CHOL_CACHE_TOL=5`` and
``GB_INFOMAT_PER_BLOCK=1``.

Public entry points: :func:`chol_cache_enabled`, :func:`source_uid_enabled`,
:func:`new_source_uids` (the per-source IDs issued by the GB moves) and
:func:`get_chol_cache`.

What is cached
--------------
Every GB in-model block draws its jumps from a per-source factor ``B`` of the
inverse information matrix in the sampling basis (``B B^T = Gamma^-1``, or the
eigen-axis table when ``GB_INMODEL_EIGEN_AXIS`` is ready), built by
:meth:`GBSpecialBase._compute_proposal_cholesky` and held fixed for the whole
block. Computing it for every source of every block dominated
``inmodel_cholesky``. With ``GB_CHOL_CACHE=1`` the factor of each living
source is computed once per refresh and served to every later block that
contains the source, until the next refresh.

With the observable eigen proposal on (``GB_INMODEL_OBSERVABLE_EIGEN``), the
same computation also stashes the observable-basis information matrix
``Gamma_z`` on the move (``move._obs_gamma_z``, indexed by sorter position).
It is cached beside the factor and written back into that stash on every hit,
so ``_observable_eigen_prepare`` reads what a direct computation would have
left there.

Residency
---------
Entries live in HOST (numpy) memory, one append-only table per walker
(:class:`_WalkerTable`). A block gathers its rows on the host and uploads a
single ``(n, ndim, ndim)`` array to the move's device (``move.xp``); nothing
is held on the GPU between blocks.

Keying: per-source IDs
----------------------
Sorter positions are rebuilt every propose, so an entry is found by what the
source is. Every living GB source carries a persistent integer ID (int64,
``-1`` = none): ``GBState.source_uid`` on the GB sub-state, one per leaf slot
next to ``d_h``/``h_h``, and ``BandSorter.source_uid`` per sorter row,
gathered exactly like ``coords``. An entry is keyed by ``(walker, ID)``:
looked up within the source's own walker (the factor depends on that
walker's noise) and never by its temperature rung. The ID stays attached to
its source through every path that moves or relabels a GB leaf:

* sorter build: alive rows take the sub-state's IDs; an alive row without
  one (the first use after a load or relaunch, the reseeded hot rung) gets a
  fresh ID there, before any lookup (``GBSpecialBase._issue_missing_uids``);
* RJ birth: a new ID; death: the ID is retired (``_rj_flip_uids``);
  replace: a new ID (``_replace_uids``) -- it is a different source;
* in-model moves change coords, not rows: the ID stays;
* vertical (all-rungs) and tempering swaps relabel a ROW's rung/walker
  (``BandSorter.exchange_cell_labels*``): the ID is a column of that row, so
  it moves with the source by construction;
* the write-back repack places each row's ID where its coords land
  (``GBSpecialBase._write_back_uids``);
* GFState copies, the multi-rank walker slices and merges, the head's block
  merge and the one-walker replica merge carry the column like ``inds``;
* the cold->hot reseed gives the reseeded rung no ID (they are new sources).

IDs are unique for the lifetime of the process and, with overwhelming
probability, across ranks and restarts (:func:`new_source_uids`: a random
per-process prefix over a counter). They are NOT stored in the HDF5 store
(``GBState.storage_arrays`` does not list them); a mid-iteration checkpoint
pickles the state and so keeps them, which is harmless. The ID of a source
that changes walker (a tempering walker permutation) is a new key in the new
walker: computed once there.

Fallback: coordinate matching. A row WITHOUT an ID (a sorter built without
the column: ``GB_CHOL_CACHE_UID=0``, or any caller that passes none) takes
the pre-ID path. A source is matched only within its own walker; the
candidates are the ``GB_CHOL_CACHE_WINDOW`` entries on each side of the
source in f0 (wide enough to span every rung's copy of the same source); the
best candidate is the one with the smallest worst-column distance over
sampling columns 0-2 (amplitude or distance, f0, fdot or Mc), each column
measured in that entry's own marginal width ``sqrt(diag(B B^T))``. It is a
hit when that distance is at most ``GB_CHOL_CACHE_TOL``. A hit moves the
entry's key to the source's current coordinates, so the key follows the
source as it random-walks. Its two failure modes -- two sources of one
walker within the tolerance can trade factors, and a source carried beyond
the tolerance between blocks by another move misses -- do not exist for a
row with an ID.

Refresh (global ticker)
-----------------------
Each cache keeps one ticker: the largest propose index (``move.time``) seen
from any move that shares it, so a slower move never rewinds it. When the
ticker enters a new multiple of ``GB_CHOL_CACHE_EVERY``, the first in-model
block of the next proposal (:meth:`GBSpecialBase._ensure_proposal_tables`)
rebuilds the whole cache from every alive source of the sorter (all walkers,
all rungs) in ``GB_CHOL_CACHE_BATCH``-sized calls. A refresh belongs to no block and has no
buffer slots, so it takes the engine's slot-free route: the lookup Gram
``<dh_a|dh_b>`` with ``SIGHET_INFOMAT_ENGINE=lookup`` (production), otherwise
the chunked delegate (slow; ``_infomat_route_check`` reports the cost).

Births and misses
-----------------
Rows of a block that find no entry (accepted births and replacements -- a new
ID --, a source that changed walker, a walker with no entries; without an ID
also sources that moved beyond tolerance) are computed in one call per
block, before the repeats, through the block's own route -- with their
buffer slots, so the sig-het fast route applies -- and appended to the cache
under their IDs. The factor is fixed for the whole block exactly as with
direct factors, so the proposal stays symmetric: a borrowed or slightly
stale factor costs acceptance rate, never detailed balance.

End-of-block retrack (rows without an ID only)
----------------------------------------------
The ~25 repeats of a block can carry a source several tolerances away from
where the block found it (Gram steps are large). After the final write-back,
``GBSpecialBase._run_in_model_repeats`` calls :meth:`_CholCache.retrack` with
the final coordinates, so the next block finds a coordinate-keyed entry. An
ID-keyed entry needs no retrack and keeps the key it was stored with.

Failure
-------
Any exception in a refresh, a take or a retrack disables that cache for the
rest of the run (one WARNING with the traceback) and the move falls back to
computing every block's factors directly, as if the cache were off.

The lookup Gram engine came in together with the cache, so the failure path
also unsets ``SIGHET_INFOMAT_ENGINE`` and the direct factors go back to the
validated route (sig-het second differences with slots). This has to be an
environment change: GBGPU's ``GBSignalHetComputations.information_matrix``
reads ``SIGHET_INFOMAT_ENGINE`` from ``os.environ`` on every call and has no
argument or attribute that selects the route. The change is process-wide, so
it is logged once at WARNING; it also applies to every other move and every
other cache on the rank (their later refreshes go through the chunked
delegate).

Knobs (environment)
-------------------
``GB_CHOL_CACHE`` (default ``0``)
    ``1`` turns the cache on; read by the move's gate on every call.
``GB_CHOL_CACHE_EVERY`` (default ``40``)
    Refresh period in proposes (40 proposes x ~25 repeats is ~1000 in-model
    steps). Minimum 1.
``GB_CHOL_CACHE_TOL`` (default ``5``)
    Coordinate-fallback match tolerance, in the entry's marginal widths.
``GB_CHOL_CACHE_BATCH`` (default ``4096``)
    Sources per factor call during a refresh (bounds peak device memory).
    Minimum 1.
``GB_CHOL_CACHE_WINDOW`` (default ``32``)
    Coordinate fallback: candidate entries examined on each side of a source
    in f0. Minimum 1.
``GB_CHOL_CACHE_UID`` (default ``1``)
    ``0`` is the rollback switch: no sorter carries IDs (the GB moves skip
    every ID step and write ``-1`` back), so every row takes the coordinate
    path, exactly as before the IDs. Read at every sorter build
    (:func:`source_uid_enabled`), like ``GB_CHOL_CACHE``.

``GB_CHOL_CACHE_EVERY``, ``_TOL``, ``_BATCH`` and ``_WINDOW`` are read once,
when a cache is created (:class:`_CholCache` ``__init__``).

Logging
-------
Every line reads ``<move name>: [GB_CHOL_CACHE] ...`` (the branch name when
no move is in hand):

* ``refreshed N sources in T s (...)`` -- INFO, each refresh (ticker, period,
  info-matrix engine, host MB);
* ``H hits / M misses over the last 500 blocks (...)`` -- INFO; the counts
  cover every move that shares the cache;
* ``DISABLED after an error in <refresh|take|retrack> ...`` -- WARNING with
  the traceback, at most once per cache;
* ``unset SIGHET_INFOMAT_ENGINE=... for the rest of the run`` -- WARNING, once
  per process.

Registry, deepcopy and pickle
-----------------------------
There is one cache per process (one MPI rank) per key ``(branch, observable
eigen mode, eigen-axis flag)``, held in the module-level ``_CHOL_CACHES`` and
reached through :func:`get_chol_cache`. The mode and flag are in the key
because they change what ``_compute_proposal_cholesky`` returns (a joint
factor or an eigen-axis table, with or without ``Gamma_z``), so those factors
are not interchangeable. Every GB move of a branch shares one map, so a
refresh is paid once per ticker period, not once per move.

The registry is module-level on purpose, rather than an attribute of a move.
The cache is per-process runtime state whose host arrays scale with the live
source count, so a move that is deep-copied or pickled carries no reference
to it and stays light (settings-tree objects must survive
``pickle.loads(pickle.dumps(copy.deepcopy(obj)))``). The cache itself holds
only numpy arrays and plain Python values -- never an array module; the
device module comes from ``move.xp`` at call time -- so it can be deep-copied
and pickled too.
"""

from __future__ import annotations

import logging
import os
import time
from typing import NamedTuple, Optional

import numpy as np

from ...utils.utility import asnumpy, get_array_module

logger = logging.getLogger(__name__)

__all__ = ["chol_cache_enabled", "get_chol_cache", "new_source_uids",
           "source_uid_enabled"]

#: Defaults of the ``GB_CHOL_CACHE_*`` knobs (see the module docstring).
DEFAULT_EVERY = 40
DEFAULT_TOL = 5.0
DEFAULT_BATCH = 4096
DEFAULT_WINDOW = 32

#: Sampling column the per-walker tables are sorted on (f0).
_F0_COL = 1
#: Sampling columns compared against the tolerance (amplitude or distance, f0,
#: fdot or Mc).
_MATCH_COLS = np.array([0, 1, 2])
#: Smallest row capacity a walker table is allocated with.
_MIN_CAPACITY = 1024
#: Blocks between two hits/misses report lines.
_REPORT_EVERY_BLOCKS = 500
#: GBGPU's info-matrix route selector, unset when a cache fails.
_INFOMAT_ENGINE_ENV = "SIGHET_INFOMAT_ENGINE"

#: Per-process registry: one :class:`_CholCache` per key (see the module
#: docstring for why this is module-level rather than on the move).
_CHOL_CACHES: dict = {}

#: Source IDs are ``prefix << _UID_COUNTER_BITS | counter``: the prefix is
#: drawn once per process (31 random bits from ``os.urandom``, never from a
#: sampler RNG stream), the counter counts the IDs that process issued.
_UID_COUNTER_BITS = 32
_UID_STATE = {"pid": None, "prefix": 0, "next": 0}


def chol_cache_enabled() -> bool:
    """Return whether the proposal-factor cache is switched on.

    Reads ``GB_CHOL_CACHE`` from the environment on every call (default
    ``"0"``; only ``"1"`` turns it on). Called by
    ``GBSpecialBase._chol_cache`` and by the per-source ID hooks of the GB
    moves.

    Returns:
        bool: True when ``GB_CHOL_CACHE=1``.
    """
    return os.environ.get("GB_CHOL_CACHE", "0") == "1"


def source_uid_enabled() -> bool:
    """Return whether GB sorters carry per-source IDs.

    Reads ``GB_CHOL_CACHE_UID`` on every call (default ``"1"``; ``"0"`` is
    the rollback switch to coordinate matching, see the module docstring).
    The GB moves consult it together with :func:`chol_cache_enabled` when
    they build a sorter.

    Returns:
        bool: True unless ``GB_CHOL_CACHE_UID=0``.
    """
    return os.environ.get("GB_CHOL_CACHE_UID", "1") != "0"


def new_source_uids(n: int) -> np.ndarray:
    """Issue ``n`` fresh per-source IDs.

    Unique for the lifetime of the process. Across the ranks of a run (each
    issues its own births) and across restarts (a mid-iteration checkpoint
    brings the previous process's IDs back) they differ by their random
    prefix: two processes collide only if they drew the same 31-bit prefix.
    The prefix is redrawn in a forked child and when the counter runs out.
    The prefix comes from ``os.urandom``, so drawing IDs never consumes a
    sampler random stream. Called by the GB moves at sorter build
    (``_issue_missing_uids``), RJ births (``_rj_flip_uids``) and accepted
    replacements (``_replace_uids``).

    Args:
        n: Number of IDs to issue (cast to ``int``).

    Returns:
        np.ndarray: ``(n,)`` host int64 array of consecutive IDs, all ``>= 0``
        (``prefix << 32 | counter``).
    """
    n = int(n)
    st = _UID_STATE
    if st["pid"] != os.getpid() or st["next"] + n > (1 << _UID_COUNTER_BITS):
        st["pid"] = os.getpid()
        st["prefix"] = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
        st["next"] = 0
    start = (st["prefix"] << _UID_COUNTER_BITS) + st["next"]
    st["next"] += n
    return np.arange(start, start + n, dtype=np.int64)


def get_chol_cache(key) -> "_CholCache":
    """Return this process's cache for ``key``, creating it on first use.

    The cache lives in the module-level registry ``_CHOL_CACHES`` (one per
    process and key), never on a move. Called by ``GBSpecialBase._chol_cache``.

    Args:
        key (tuple): ``(branch name, observable eigen mode, eigen-axis flag)``,
            e.g. ``("gb", "axis", False)``.

    Returns:
        _CholCache: The cache for ``key``. A new cache reads
        ``GB_CHOL_CACHE_EVERY`` / ``_TOL`` / ``_BATCH`` / ``_WINDOW`` once.
    """
    cache = _CHOL_CACHES.get(key)
    if cache is None:
        cache = _CHOL_CACHES[key] = _CholCache(key)
    return cache


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Integer knob; empty or unset means ``default``; clamped to ``minimum``."""
    return max(int(os.environ.get(name, str(default)) or default), minimum)


def _env_float(name: str, default: float) -> float:
    """Float knob; empty or unset means ``default``."""
    return float(os.environ.get(name, str(default)) or default)


def _take_rows(a, rows):
    """``a[rows]`` for a host (numpy) row index, whether ``a`` is numpy or cupy."""
    if isinstance(a, np.ndarray):
        return a[rows]
    return a[get_array_module(a).asarray(rows)]


def _regrow(old, n_filled: int, capacity: int, template, fill=None):
    """A new ``capacity``-row buffer holding the first ``n_filled`` rows of ``old``.

    Row shape and dtype come from ``old``, or from ``template`` when there is
    no ``old`` yet. The other rows are uninitialised, or ``fill`` if given.
    """
    like = template if old is None else old
    shape = (capacity,) + like.shape[1:]
    if fill is None:
        buf = np.empty(shape, dtype=like.dtype)
    else:
        buf = np.full(shape, fill, dtype=like.dtype)
    if old is not None:
        buf[:n_filled] = old[:n_filled]
    return buf


class _WalkerTable:
    """One walker's cache entries: append-only host arrays, ID and f0 indexes.

    Rows ``[0, n)`` are filled. They are appended with amortized doubling and
    never move, so an entry number recorded for a block stays valid until a
    refresh replaces the table. ``coords`` holds each entry's coordinate key
    (the coordinates it was last matched or retracked at) and is the only
    array rewritten in place; ``uids`` is fixed at append.

    Attributes:
        n: Number of filled rows.
        coords: ``(capacity, ndim)`` key coordinates, sampling basis.
        factors: ``(capacity, ndim, ndim)`` proposal factors ``B``.
        widths: ``(capacity, 3)`` marginal widths of columns ``_MATCH_COLS``
            in coordinate units.
        uids: ``(capacity,)`` source ID each entry was stored under, ``-1``
            for a coordinate-keyed entry.
        gamma_z: ``(capacity, nz, nz)`` observable information matrices, or
            ``None`` while no row carries one. Rows stored without one are NaN
            (the move treats a non-finite ``Gamma_z`` as "no eigen table").
        uid_order: Filled rows sorted by ID.
        uid_sorted: The IDs in that order.
        f0_order: Filled rows sorted by key f0.
        f0_sorted: The key f0 values in that order.
    """

    def __init__(self):
        self.n = 0
        self.coords = None
        self.factors = None
        self.widths = None
        self.uids = None
        self.gamma_z = None
        self.uid_order = None
        self.uid_sorted = None
        self.f0_order = None
        self.f0_sorted = None

    def append(self, coords, factors, widths, gamma_z, uids=None) -> np.ndarray:
        """Append rows and return their entry numbers.

        Args:
            coords: ``(k, ndim)`` key coordinates.
            factors: ``(k, ndim, ndim)`` proposal factors.
            widths: ``(k, 3)`` marginal widths.
            gamma_z: ``(k, nz, nz)`` observable information matrices, or
                ``None`` (stored as NaN if the table carries ``Gamma_z``).
            uids: ``(k,)`` source IDs (``-1`` = none), or ``None`` (all none).
        """
        n, k = self.n, len(coords)
        uids = (np.full(k, -1, dtype=np.int64) if uids is None
                else np.asarray(uids, dtype=np.int64))
        if self.coords is None or n + k > len(self.coords):
            capacity = max(2 * (n + k), _MIN_CAPACITY)
            self.coords = _regrow(self.coords, n, capacity, coords)
            self.factors = _regrow(self.factors, n, capacity, factors)
            self.widths = _regrow(self.widths, n, capacity, widths)
            self.uids = _regrow(self.uids, n, capacity, uids)
            if self.gamma_z is not None:
                self.gamma_z = _regrow(self.gamma_z, n, capacity, None, fill=np.nan)
        self.coords[n:n + k] = coords
        self.factors[n:n + k] = factors
        self.widths[n:n + k] = widths
        self.uids[n:n + k] = uids
        if gamma_z is not None:
            if self.gamma_z is None:
                # the first rows with Gamma_z: every earlier row reads as NaN
                self.gamma_z = _regrow(None, n, len(self.coords), gamma_z, fill=np.nan)
            self.gamma_z[n:n + k] = gamma_z
        # rows appended without Gamma_z keep the NaN fill
        self.n = n + k
        self.reindex()
        self.uid_order = np.argsort(self.uids[:self.n], kind="stable")
        self.uid_sorted = self.uids[:self.n][self.uid_order]
        return np.arange(n, n + k)

    def find(self, uids) -> np.ndarray:
        """Entry number stored under each ID, ``-1`` where there is none."""
        uids = np.asarray(uids, dtype=np.int64)
        if self.n == 0 or uids.size == 0:
            return np.full(uids.shape, -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.uid_sorted, uids), self.n - 1)
        return np.where(self.uid_sorted[pos] == uids, self.uid_order[pos], -1)

    def reindex(self):
        """Rebuild the f0 sort index after keys were added or moved."""
        f0 = self.coords[:self.n, _F0_COL]
        self.f0_order = np.argsort(f0, kind="stable")
        self.f0_sorted = f0[self.f0_order]

    def nbytes(self) -> int:
        """Host bytes allocated by the table's arrays (capacity, not fill)."""
        return sum(a.nbytes for a in (self.coords, self.factors, self.widths,
                                      self.uids, self.gamma_z) if a is not None)


class _Lookup(NamedTuple):
    """Result of :meth:`_CholCache.match` for one block's rows.

    Attributes:
        hit: ``(n,)`` bool, row found an entry.
        walkers: ``(n,)`` walker of each row (the table it was looked up in).
        entries: ``(n,)`` entry number within that table, ``-1`` on a miss.
        factors: ``(n, ndim, ndim)`` host factors; zeros on misses.
        gamma_z: ``(n, nz, nz)`` host ``Gamma_z`` (NaN where absent), or
            ``None`` when no hit came from a table that carries one.
    """

    hit: np.ndarray
    walkers: np.ndarray
    entries: np.ndarray
    factors: np.ndarray
    gamma_z: Optional[np.ndarray]


class _CholCache:
    """Host-resident in-model proposal factors, mapped to living sources.

    See the module docstring for the design (keying, refresh ticker, misses,
    retrack, failure) and the knobs. Instances come from
    :func:`get_chol_cache`; the GB move's ``_chol_cache`` gate decides
    whether one is used at all.

    Attributes:
        key: ``(branch, observable eigen mode, eigen-axis flag)``.
        every: Refresh period in proposes (``GB_CHOL_CACHE_EVERY``).
        tol: Match tolerance in marginal widths (``GB_CHOL_CACHE_TOL``).
        batch: Sources per factor call in a refresh (``GB_CHOL_CACHE_BATCH``).
        window: Candidates on each side in f0 (``GB_CHOL_CACHE_WINDOW``).
        tick: Largest propose index seen (the global ticker).
        epoch: ``tick // every`` at the last refresh; ``None`` before the
            first one (the cache serves nothing until then).
        tables: ``{walker: _WalkerTable}``.
        disabled: Set by the first failure; permanent.
    """

    def __init__(self, key):
        self.key = key
        self.every = _env_int("GB_CHOL_CACHE_EVERY", DEFAULT_EVERY)
        self.tol = _env_float("GB_CHOL_CACHE_TOL", DEFAULT_TOL)
        self.batch = _env_int("GB_CHOL_CACHE_BATCH", DEFAULT_BATCH)
        self.window = _env_int("GB_CHOL_CACHE_WINDOW", DEFAULT_WINDOW)
        self.tick = -1
        self.epoch = None
        self.tables = {}
        self.disabled = False
        # (ids, walkers, entries) of the block the last take served; consumed
        # by retrack at the end of that block.
        self._served = None
        self._hits_since_report = 0
        self._misses_since_report = 0
        self._blocks_served = 0

    # ---- ticker and refresh ------------------------------------------------

    def due(self, propose_index) -> bool:
        """Advance the ticker to ``propose_index`` and say if a refresh is due.

        The ticker only moves forward (moves sharing the cache count proposes
        separately). A refresh is due when the cache has never been refreshed
        or the ticker has entered a new ``every``-period since the last one;
        never once the cache is disabled.

        Args:
            propose_index (int): The calling move's propose counter
                (``move.time``).

        Returns:
            bool: True when :meth:`refresh` should run before the next block.
        """
        self.tick = max(self.tick, int(propose_index))
        return (not self.disabled
                and (self.epoch is None or self.tick // self.every != self.epoch))

    def refresh(self, move, model, band_sorter):
        """Rebuild the whole cache from every alive source of ``band_sorter``.

        Computes the factors in ``batch``-sized calls through the move's
        slot-free route and replaces all tables. On error the cache is
        disabled (see :meth:`_fail`). Logs one ``refreshed N sources`` INFO
        line.

        Args:
            move (GBSpecialBase): The GB move; supplies ``xp``, ``name`` and
                ``_compute_proposal_cholesky``.
            model (object): The eryn model passed to the move's propose.
            band_sorter (BandSorter): This propose's sorter; every row with
                ``inds`` True (all walkers, all rungs) is cached.
        """
        if self.disabled:
            return
        t0 = time.perf_counter()
        try:
            alive = move.xp.where(band_sorter.inds)[0]
            self.tables = {}
            self._served = None
            n_alive = int(alive.shape[0])
            for start in range(0, n_alive, self.batch):
                ids = alive[start:start + self.batch]
                factors = move._compute_proposal_cholesky(model, band_sorter, ids)
                self._store(move, band_sorter, ids, factors)
        except Exception as exc:  # noqa: BLE001 -- see _fail
            self._fail(move, "refresh", exc)
            return
        self.epoch = self.tick // self.every
        logger.info(
            "%s: [GB_CHOL_CACHE] refreshed %d sources in %.1f s (propose %d, "
            "every %d proposes, %s=%s, %.1f MB host)", move.name, n_alive,
            time.perf_counter() - t0, self.tick, self.every, _INFOMAT_ENGINE_ENV,
            os.environ.get(_INFOMAT_ENGINE_ENV, "") or "default",
            self.nbytes() / 2**20)

    # ---- serving a block ---------------------------------------------------

    def take(self, move, model, band_sorter, ids, slots, buffer_obj):
        """Proposal factors for one block's sources ``ids`` (on ``move.xp``).

        Hits are served from the cache; misses are computed in one call
        through the block's own route (``slots`` and ``buffer_obj`` as for
        :meth:`GBSpecialBase._compute_proposal_cholesky`) and stored. Also
        leaves ``move._proposal_param_scales`` set and, for hits, writes the
        cached ``Gamma_z`` into ``move._obs_gamma_z`` -- the two side effects
        the direct computation has. Before the first refresh, once disabled,
        or on error (which disables), the whole block is computed directly.

        Args:
            move (GBSpecialBase): The GB move.
            model (object): The eryn model passed to the move's propose.
            band_sorter (BandSorter): This propose's sorter.
            ids (xp.ndarray): Sorter row ids of the block's alive sources.
            slots (xp.ndarray or None): Buffer slot of each row of ``ids``
                (``None`` = no slots, the slot-free route).
            buffer_obj (SubBandBuffer or None): The bound sub-band buffer,
                forwarded as the router's ``slot_holder``.

        Returns:
            xp.ndarray: ``(len(ids), ndim, ndim)`` proposal factors ``B`` on
            ``move.xp``, row-aligned with ``ids``.
        """
        self._served = None
        if self.disabled or self.epoch is None:
            return move._compute_proposal_cholesky(
                model, band_sorter, ids, slots=slots, buffer_obj=buffer_obj)
        xp = move.xp
        try:
            found = self.match(band_sorter, ids)
            chol = xp.asarray(found.factors)
            miss = np.nonzero(~found.hit)[0]
            if miss.size:
                # Computed before the Gamma_z write-back below: the direct
                # computation may replace move._obs_gamma_z with a new array.
                miss_ids = _take_rows(ids, miss)
                miss_factors = move._compute_proposal_cholesky(
                    model, band_sorter, miss_ids,
                    slots=None if slots is None else _take_rows(slots, miss),
                    buffer_obj=buffer_obj)
                found.entries[miss] = self._store(
                    move, band_sorter, miss_ids, miss_factors)
                chol[xp.asarray(miss)] = miss_factors
            else:
                # no direct computation ran: set its scale side effect here
                move._proposal_param_scales = self._param_scales(move, band_sorter)
            if found.gamma_z is not None and found.hit.any():
                self._write_gamma_z(move, band_sorter, ids, found)
            # retrack only the coordinate-keyed rows: an ID entry keeps its key
            coord_entries = np.where(
                self._uids(band_sorter, ids, len(found.hit)) >= 0, -1,
                found.entries)
            self._served = (asnumpy(ids).copy(), found.walkers, coord_entries)
        except Exception as exc:  # noqa: BLE001 -- see _fail
            self._fail(move, "take", exc)
            return move._compute_proposal_cholesky(
                model, band_sorter, ids, slots=slots, buffer_obj=buffer_obj)
        self._count(move, int(found.hit.sum()), int(miss.size))
        return chol

    def match(self, band_sorter, ids) -> _Lookup:
        """Find each source's entry in the table of its walker.

        A row with a source ID takes the entry stored under that ID, if any
        (a miss otherwise -- never a neighbour's entry). A row without one
        takes the coordinate fallback: of the ``2 * window`` entries nearest
        in f0, the one with the smallest worst-column distance over
        ``_MATCH_COLS`` in that entry's marginal widths; a hit is a distance
        ``<= tol``, and the hit entry's key is set to the row's current
        coordinates.

        Args:
            band_sorter (BandSorter): This propose's sorter (``coords``,
                ``walker_inds`` and, when present, ``source_uid`` are read).
            ids (xp.ndarray): Sorter row ids to look up.

        Returns:
            _Lookup: Host-side hit mask, walkers, entry numbers, factors and
            ``Gamma_z`` for the rows of ``ids``.
        """
        coords = asnumpy(band_sorter.coords[ids])
        n_rows, ndim = coords.shape
        uids = self._uids(band_sorter, ids, n_rows)
        hit = np.zeros(n_rows, dtype=bool)
        entries = np.full(n_rows, -1, dtype=np.int64)
        factors = np.zeros((n_rows, ndim, ndim))
        gamma_z = None
        walkers = self._walkers(band_sorter, ids)
        offsets = np.arange(-self.window, self.window)
        for walker in np.unique(walkers):
            table = self.tables.get(int(walker))
            if table is None or table.n == 0:
                continue
            in_walker = np.nonzero(walkers == walker)[0]
            # rows with an ID: the entry stored under it, nothing else
            keyed = in_walker[uids[in_walker] >= 0]
            found = table.find(uids[keyed])
            hit_rows, hit_entries = [keyed[found >= 0]], [found[found >= 0]]
            # rows without one: the coordinate fallback (the pre-ID path)
            rows = in_walker[uids[in_walker] < 0]
            if rows.size:
                query = coords[rows]
                # the 2 * window table positions around each row's f0
                # insertion point, clipped to the filled range
                pos = np.searchsorted(table.f0_sorted, query[:, _F0_COL])
                cand = table.f0_order[
                    np.clip(pos[:, None] + offsets[None, :], 0, table.n - 1)]
                with np.errstate(divide="ignore", invalid="ignore"):
                    dist = (np.abs(query[:, None, _MATCH_COLS]
                                   - table.coords[cand][:, :, _MATCH_COLS])
                            / table.widths[cand])
                # a zero width gives inf or NaN: never a match
                score = np.nan_to_num(dist.max(-1), nan=np.inf)
                best = score.argmin(1)
                r = np.arange(len(rows))
                best_entry = cand[r, best]
                ok = score[r, best] <= self.tol
                if ok.any():
                    hit_rows.append(rows[ok])
                    hit_entries.append(best_entry[ok])
                    table.coords[best_entry[ok]] = query[ok]
                    table.reindex()
            hit_rows = np.concatenate(hit_rows)
            hit_entries = np.concatenate(hit_entries)
            if not hit_rows.size:
                continue
            hit[hit_rows] = True
            entries[hit_rows] = hit_entries
            factors[hit_rows] = table.factors[hit_entries]
            if table.gamma_z is not None:
                if gamma_z is None:
                    gamma_z = np.full((n_rows,) + table.gamma_z.shape[1:], np.nan)
                gamma_z[hit_rows] = table.gamma_z[hit_entries]
        return _Lookup(hit, walkers, entries, factors, gamma_z)

    def retrack(self, ids, coords):
        """End of block: move the served entries' keys to the final coordinates.

        Coordinate-keyed entries only (rows served without a source ID); an
        ID entry keeps its key. The key is set when the block starts
        (:meth:`match`); the block's repeats then move the source, often
        beyond tolerance. Applies only
        to the block the last :meth:`take` served: ``ids`` must be that
        block's ids in the same order, else nothing changes. ``coords`` are
        the final sampling coordinates, row-aligned with ``ids``. On error
        the cache is disabled (see :meth:`_fail`). Called by
        ``GBSpecialBase._run_in_model_repeats`` after the block's final
        write-back.

        Args:
            ids (xp.ndarray): The block's sorter row ids, in the order
                :meth:`take` received them.
            coords (xp.ndarray): ``(len(ids), ndim)`` final sampling-basis
                coordinates.
        """
        served, self._served = self._served, None
        if served is None or self.disabled:
            return
        served_ids, walkers, entries = served
        ids_now = asnumpy(ids)
        if ids_now.shape != served_ids.shape or not np.array_equal(ids_now, served_ids):
            return
        try:
            final = asnumpy(coords)
            for walker in np.unique(walkers):
                table = self.tables.get(int(walker))
                rows = (walkers == walker) & (entries >= 0)
                if table is None or not rows.any():
                    continue
                table.coords[entries[rows]] = final[rows]
                table.reindex()
        except Exception as exc:  # noqa: BLE001 -- see _fail
            self._fail(None, "retrack", exc)

    # ---- internals ---------------------------------------------------------

    def _store(self, move, band_sorter, ids, factors) -> np.ndarray:
        """Append freshly computed ``factors`` for ``ids``; return entry numbers.

        Must run right after ``move._compute_proposal_cholesky`` for the same
        ``ids``: the marginal widths use the scales and the ``Gamma_z`` comes
        from the stash that call left on the move.
        """
        coords = asnumpy(band_sorter.coords[ids])
        factors = asnumpy(factors)
        gamma_z = self._gamma_z_of(move, ids)
        widths = self._marginal_widths(move, factors)
        walkers = self._walkers(band_sorter, ids)
        uids = self._uids(band_sorter, ids, len(walkers))
        entries = np.empty(len(walkers), dtype=np.int64)
        for walker in np.unique(walkers):
            rows = walkers == walker
            table = self.tables.get(int(walker))
            if table is None:
                table = self.tables[int(walker)] = _WalkerTable()
            entries[rows] = table.append(
                coords[rows], factors[rows], widths[rows],
                None if gamma_z is None else gamma_z[rows], uids[rows])
        return entries

    @staticmethod
    def _uids(band_sorter, ids, n_rows) -> np.ndarray:
        """Host source ID of each row (``-1`` where there is none).

        Every row of a sorter that carries no ID column reads ``-1``.
        """
        col = getattr(band_sorter, "source_uid", None)
        if col is None:
            return np.full(n_rows, -1, dtype=np.int64)
        return asnumpy(col[ids]).astype(np.int64)

    @staticmethod
    def _walkers(band_sorter, ids) -> np.ndarray:
        """Host walker index of each source: the lookup table it belongs to.

        Deliberately not the rung, which a vertical swap relabels.
        """
        return asnumpy(band_sorter.walker_inds[ids]).astype(np.int64)

    @staticmethod
    def _marginal_widths(move, factors) -> np.ndarray:
        """``sqrt(diag(B B^T))`` on ``_MATCH_COLS``, in coordinate units.

        ``B`` is in the conditioned basis ``y = x / s``
        (``move._proposal_param_scales``), so the widths are scaled by ``s``.
        """
        scales = asnumpy(move._proposal_param_scales)[_MATCH_COLS]
        return np.sqrt((factors[:, _MATCH_COLS, :] ** 2).sum(-1)) * scales[None, :]

    @staticmethod
    def _gamma_z_of(move, ids):
        """Host ``Gamma_z`` the move just stashed for ``ids``, or ``None``."""
        stash = getattr(move, "_obs_gamma_z", None)
        if stash is None or move._obs_eigen_mode() == "off":
            return None
        return asnumpy(stash[ids])

    @staticmethod
    def _param_scales(move, band_sorter):
        """The conditioning scales ``_compute_proposal_cholesky`` would set."""
        scales = move.xp.ones(band_sorter.coords.shape[1])
        if move._fdot_col is not None:
            scales[move._fdot_col] = move._fdot_scale
        return scales

    @staticmethod
    def _write_gamma_z(move, band_sorter, ids, found: _Lookup):
        """Write the hits' cached ``Gamma_z`` into ``move._obs_gamma_z``.

        The stash is indexed by sorter position; it is replaced by a NaN array
        of the current sorter size when missing or mis-shaped, as the direct
        computation does.
        """
        xp = move.xp
        n_src = int(band_sorter.inds.shape[0])
        nz = int(found.gamma_z.shape[-1])
        stash = getattr(move, "_obs_gamma_z", None)
        if (stash is None or int(stash.shape[0]) != n_src
                or int(stash.shape[-1]) != nz):
            stash = xp.full((n_src, nz, nz), xp.nan)
        hit_rows = np.nonzero(found.hit)[0]
        stash[xp.asarray(_take_rows(ids, hit_rows))] = xp.asarray(found.gamma_z[found.hit])
        move._obs_gamma_z = stash

    def _count(self, move, hits: int, misses: int):
        """Tally a served block; log hits/misses every ``_REPORT_EVERY_BLOCKS``."""
        self._hits_since_report += hits
        self._misses_since_report += misses
        self._blocks_served += 1
        if self._blocks_served % _REPORT_EVERY_BLOCKS:
            return
        total = self._hits_since_report + self._misses_since_report
        logger.info(
            "%s: [GB_CHOL_CACHE] %d hits / %d misses over the last %d blocks "
            "(%.1f %% hits, %.1f MB host)", move.name, self._hits_since_report,
            self._misses_since_report, _REPORT_EVERY_BLOCKS,
            100.0 * self._hits_since_report / max(total, 1), self.nbytes() / 2**20)
        self._hits_since_report = self._misses_since_report = 0

    def _fail(self, move, where: str, exc: BaseException):
        """Disable the cache for the rest of the run after an error.

        The run must not stop for a cache fault: the fallback is the direct
        per-block computation the cache replaced, hence the broad ``except``
        at the call sites. Also unsets ``SIGHET_INFOMAT_ENGINE`` (process
        wide, logged once; see the module docstring for why it has to be the
        environment). ``move`` is ``None`` when called from :meth:`retrack`.
        """
        self.disabled = True
        self._served = None
        # retrack has no move in hand: name the cache's branch instead
        name = move.name if move is not None else str(self.key[0])
        logger.warning(
            "%s: [GB_CHOL_CACHE] DISABLED after an error in %s (%r); every "
            "block computes its own factors from here on (cache %r).",
            name, where, exc, self.key, exc_info=exc)
        engine = os.environ.pop(_INFOMAT_ENGINE_ENV, None)
        if engine is not None:
            logger.warning(
                "%s: [GB_CHOL_CACHE] unset %s=%r for the rest of the run, "
                "process-wide: direct factors go back to the sig-het "
                "second-difference route (slot-free calls to the chunked "
                "delegate).", name, _INFOMAT_ENGINE_ENV, engine)

    def nbytes(self) -> int:
        """Host bytes allocated by all tables."""
        return sum(t.nbytes() for t in self.tables.values())
