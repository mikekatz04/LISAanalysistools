"""GB warm-start SELF refit: the run refits its warm-start mixture from its own store.

USER RULINGS (Mike, 2026-10-13; the stages, the cadences and the 25/75 split
were superseded by the 2026-10-14/15/16 rulings below):

* "Starting in gb search 2 (and in full pe). Let's rebuild our warm start
  proposal with the last 50 samples (x4 walkers)."
* "We should keep the original warm start in the fit."
* "instead of 50-50, during and after gb refit 2 (so for full pe as well),
  sample from the previous full warm start 25% of the time and the updated
  refit warm start 75% of the time."
* "during gb search 2, the warm start refit should occur every 5
  iterations."  "full pe adaptivity: just refit every 250 iterations."
  (accepted as a periodic approximation in PE, like the residual F-stat
  refit every 150 PE iterations).

USER RULINGS (Mike, 2026-10-14/15) -- the current design:

* "Let's just build everything on the head at entry time for the refit ...
  Make that the default."  (:class:`WarmSelfHeadBuilder`; the saver hook is
  kept behind ``GB_WARM_SELF_AT_ENTRY=0``.)
* "split the refits three ways: old run refit that never changes, new refit
  with all sources, and new refit removal of sources that are already in all
  walkers ... each 1/3 of the time ... 1 refit to spread its information
  across the two self-refitting proposals" -> one build makes the FULL and
  the FILTERED (:func:`consensus_filter`) containers; PE split ``1,1,1``.
* "search split should be 100% refit on sources that do not exist in all
  four (with high confidence)" -> search split ``0,0,1``; then "do the split
  between the old run fit warm start and the refit filtered warm start
  during gb search 1/2 so we make sure not to leave out leaves from the
  previous run ... 20 % old run refit warmstart / 80% self refit warmstart
  filtered" -> search split ``0.2,0,0.8``; then (2026-10-16) "Let's split
  them 0.5/0/0.5 in search" -> ``0.5,0,0.5``.
* "Use the same refitting cadence as we do now" (search every stored row =
  every cycle at the entry), "the default refitting duration for full pe
  ... 25 iterations".
* Checked before building (9mo extracts, rows 82-89 and 103-110): ~2,160 of
  ~3,230 source groups are held by all four walkers with the same count in
  >= 95 % of rows and took 83 % of the full mixture's draws; the filtered
  mixture cuts the draws landing on a source the walker already holds from
  93 % to 57 % (useful draws ~7 % -> ~42 %); ~60 blends (held by every
  walker with differing counts) are kept by the count condition.

WHY. Per-walker GB completeness at SNR > 8 was 91 % on the 9mo run while
the UNION of the four cold walkers was 96.5 %: each walker lacked ~180-220
SNR 6-12 leaves another walker held, and the PE cold RJ acceptance
(1e-4 .. 1e-3) never closed that. A mixture fitted to the last rows of ALL
cold walkers proposes to each walker what the others hold: a cross-walker
GB exchange through the existing warm-start machinery.

THE PIECES (all in this module, wired from four places):

0. **The head build** (default; :class:`WarmSelfBuffer`,
   :class:`WarmSelfHeadBuilder`, :func:`make_head_builder` from
   ``run.prepare_main``). The head keeps the last ``GB_WARM_SELF_LAST_K``
   stored cold rows in memory (filled from the store at set-up, appended by
   the head's ``GFHDFBackend.save_step``); at a warm move's entry
   (:meth:`WarmSelfMixer.begin_propose`, called once per propose with the
   live stage) it builds when :func:`warm_self_due` says so, through the
   same pipeline as 1. below plus :func:`consensus_filter`. Every rank
   waits for it (~75 s on the 9mo). THE ADOPTION LAG (Mike, 2026-10-16:
   "refit on the current data holding our previous within-run refit
   information, use the previous information on the current proposal and
   then switch it out"; ``GB_WARM_SELF_ADOPT_LAG``, default on): the
   propose that triggered a build draws from the epoch held BEFORE it (the
   original warm start on the run's first build) and the next warm draw
   adopts the new one. On the 9mo job 767 every draw from an epoch built at
   that same entry lost 15-20 leaves and 100-527 nats per walker (a fresh
   fit sits on the current cold leaves, so the weakest die together), while
   draws on epochs >= 13 rows old were the job's best iterations.
1. **The saver build** (``GB_WARM_SELF_AT_ENTRY=0``; :class:`WarmSelfHook`,
   saver rank, ``hdfbackend``'s save
   loop). Right after one of its own saves, when the run's active recipe
   stage is ``gb_search_N`` (N >= 1) or ``full_pe`` and the stage's cadence
   has elapsed (:func:`warm_self_due`), the saver copies the last
   ``GB_WARM_SELF_LAST_K`` cold rows of ``chain/gb`` / ``inds/gb`` /
   ``log_like`` through its OWN backend's ``open("r")`` -- strictly scoped,
   closed before the hook returns, because a held read handle makes the
   saver's next ``open("a")`` fail ("file is already open for read-only",
   the 95c1734a crash; measured again on h5py 3.16 / HDF5 2.0) -- into a
   small store of the same layout, ``epoch_NNNN/leaf_rows.h5``. The parent
   build's pipeline then runs on that file exactly as
   :func:`.build.ensure_warm_start_components` runs it (fit_from_store ->
   match_referee -> referee_apply, each a plain in-process ``main(argv)``
   call through :func:`.build._default_runner`, CLI defaults for every fit
   and referee option), with the run's OWN Tobs and the saver's GPU for the
   mixture fit (``--gmm-gpu``; the whole pipeline runs inside that device's
   context, so a referee that picks its device "auto" lands on it too).
   The epoch is complete when ``DONE.json`` exists: it is written LAST,
   atomically, like the F-stat epochs'. The hook is watched the way the
   monitor hook is (:class:`WarmSelfWatchdog`): a build that holds a save
   up is reported, and one that costs stored iterations switches the
   builder off for the rest of the run.
2. **The epoch protocol** (:func:`latest_complete_epoch`,
   :func:`next_epoch_index`): ``<run_dir>/warmstart/self/epoch_NNNN/``;
   an epoch without a readable ``DONE.json`` (or whose npz is missing) is
   ignored forever, and the next build takes ``1 + max(every epoch dir)``
   so a crashed build's leftovers are never mixed into a new one.
3. **The split** (:class:`WarmSelfMixer`, the warm moves'
   ``rj_proposal_distribution``). It holds the ORIGINAL parent-derived
   container, the FULL self refit and the FILTERED one (``None`` until an
   epoch exists). Each propose DRAWS one at random with probabilities ~
   ``GB_WARM_SELF_SPLIT_SEARCH`` / ``_PE`` (``0.5,0,0.5`` = 50 % original,
   50 % filtered / ``1,1,1`` = one third each; 100 % original while no self
   epoch exists; Mike 2026-10-15: "just draw each of these with 1/3
   probability each", then for search "20 % old run refit warmstart / 80%
   self refit warmstart filtered", then 2026-10-16 "0.5/0/0.5 in search")
   and that container serves
   the birth draws AND every logpdf of the propose (the BandSorter's death
   factors are computed from the same object), so forward and reverse
   densities always agree within a propose. A fixed-probability mixture of
   exact MH kernels is itself a valid kernel.
4. **Adoption** (the head polls, every rank loads): at each propose the
   head's mixer stats the epoch root for a newer COMPLETE epoch and ships
   ``{"epoch", "choice"}`` to the compute ranks with the propose (the
   F-stat epoch directive pattern); each rank loads the epoch's two npz
   files from the shared filesystem with the original's Tobs / floor box /
   floor_eps / p_floor / circ_images / use_cupy and the per-rank seeds
   ``_warm_start_seed(gb_info, tag ^ (epoch + 1))`` (full) and the same
   XOR ``_FILTERED_SEED_XOR`` (filtered), so the candidate streams change
   per epoch and differ per rank and per container. A compute rank never
   polls: the directive's epoch is the only one it adopts, so under the
   adoption lag no rank can reach a freshly built epoch before the head
   ships it.
5. **Resume**: nothing is held in memory that the disk does not also
   hold. The head refills its buffer and the active stage's start from the
   store at set-up; both builders take the latest complete epoch's
   ``built_at_row`` from disk at every check (the saver also re-reads the
   store's recipe group at every save); the moves adopt the latest complete
   epoch at their first propose.

STORED ROWS. ``GB_WARM_SELF_LAST_K`` and both cadences count STORED ROWS
(states handed to the saver), not iterations: a numbered search stage
stores 3 rows per cycle under ``GB_SEARCH_LEGS=1`` (one per leg), so the
default 50 rows are the last ~17 search cycles, and ``EVERY_SEARCH=1`` is a
build at every warm entry, i.e. every cycle; ``full_pe`` stores one row
per iteration, so 50 rows = 50 iterations and ``EVERY_PE=25`` a build at
the first ``rj_warm_pe`` draw at least 25 iterations after the last one.

KNOBS (``GBSettings.warm_self*`` fields, env-seeded, plus the watchdog's
plain env var ``GF_MONITOR_WARN_FRAC``; code default / 9mo + 12mo
launchers). Every one needs a parent warm start
(``GB_WARM_START_COMPONENTS``); without it nothing is built:

* ``GB_WARM_SELF`` -- master switch, False / 1.
* ``GB_WARM_SELF_AT_ENTRY`` -- head build at the warm entry (True) or the
  saver build after its saves (False); True / 1.
* ``GB_WARM_SELF_EVERY_SEARCH`` -- rows between builds in ``gb_search_N``
  (N >= 1; ``<= 0`` never), 1 / 1.
* ``GB_WARM_SELF_EVERY_PE`` -- rows between builds in ``full_pe``, 25 / 25.
* ``GB_WARM_SELF_LAST_K`` -- rows x every walker fitted, 50 / 50.
* ``GB_WARM_SELF_SPLIT_SEARCH`` / ``GB_WARM_SELF_SPLIT_PE`` -- "original,
  full, filtered" weights of ``rj_warm_search`` / ``rj_warm_pe``,
  ``0.5,0,0.5`` / ``1,1,1`` on both.
* ``GB_WARM_SELF_CONSENSUS`` -- :func:`consensus_filter` threshold,
  0.95 / 0.95.
* ``GB_WARM_SELF_ADOPT_LAG`` -- the one-draw adoption lag, True / unset.
* ``GF_MONITOR_WARN_FRAC`` -- the saver build's watchdog warning fraction
  (:class:`WarmSelfWatchdog`, saver build only), 0.25 / unset.

The self epochs fit at the RUN's own Tobs and use every fit / referee /
apply option at its CLI default (the referee on the CPU unless
``GB_WARM_START_REFEREE_BACKEND`` says otherwise).

Log tag: ``[GB_WARM_SELF]`` (build, adoption, the per-propose container
choice and the per-container birth tallies, the watchdog).

See Also:
    ``docs/global-fit-navigation.md`` section 2.8 (the knob table) and
    ``recipe._warm_self_wrap`` (where the mixer is built).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
import typing

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CHOICES",
    "FILTERED_NPZ",
    "MANIFEST",
    "REFEREED_NPZ",
    "WarmSelfBuffer",
    "WarmSelfHeadBuilder",
    "WarmSelfHook",
    "WarmSelfMixer",
    "WarmSelfWatchdog",
    "build_self_epoch",
    "consensus_filter",
    "copy_leaf_rows",
    "head_builder_for",
    "latest_complete_epoch",
    "make_head_builder",
    "make_warm_self_hook",
    "next_epoch_index",
    "parse_split",
    "prune_epochs",
    "register_head_builder",
    "unregister_head_builder",
    "self_epoch_root",
    "store_stage",
    "warm_seed",
    "warm_self_cadence",
    "warm_self_due",
]

#: the completion marker, written LAST (tmp + fsync + os.replace)
MANIFEST = "DONE.json"
#: the refereed components npz the moves load
REFEREED_NPZ = "self_refereed.npz"
#: the leaf-row copy the pipeline reads (deleted once the epoch is complete)
LEAF_ROWS_H5 = "leaf_rows.h5"
#: the fitter npz (kept: small, and the referee's input)
FIT_NPZ = "self_fit.npz"
#: the FILTERED refit: the refereed npz minus the consensus clusters
FILTERED_NPZ = "self_refereed_filtered.npz"
#: XORed into the per-epoch seed tag of the FILTERED container
_FILTERED_SEED_XOR = 0x5F1E
#: XORed into the twin's tag for the head's per-propose container draw
_CHOICE_SEED_XOR = 0x2C01
_EPOCH_RE = re.compile(r"^epoch_(\d{4,})$")
#: pruning (Mike, 2026-10-14: "remove like all except the last few ... or keep
#: every 10 refits ... so we can see the difference over time"): the newest
#: KEEP_LAST_EPOCHS complete epochs and every KEEP_EVERY_EPOCH-th stay whole
KEEP_LAST_EPOCHS = 3
KEEP_EVERY_EPOCH = 10


# ---------------------------------------------------------------------------
# the epoch directory protocol
# ---------------------------------------------------------------------------
def self_epoch_root(store_path: str) -> str:
    """``<run_dir>/warmstart/self``: ONE definition for the saver and the moves.

    The run dir is the store's directory (``general_info.main_file_path`` on
    the moves, ``gb_reader.filename`` on the saver -- the same path).

    Args:
        store_path (str): The run's store h5.

    Returns:
        str: ``<dirname(abspath(store_path))>/warmstart/self``.
    """
    return os.path.join(os.path.dirname(os.path.abspath(str(store_path))),
                        "warmstart", "self")


def epoch_dir(root: str, k: int) -> str:
    """``<root>/epoch_NNNN`` (zero-padded to 4 digits) of epoch ``k``."""
    return os.path.join(root, f"epoch_{int(k):04d}")


def list_epochs(root: str) -> typing.List[int]:
    """Every ``epoch_NNNN`` directory index under ``root`` (complete or not), ascending."""
    try:
        names = os.listdir(root)
    except OSError:
        return []
    out = []
    for n in names:
        m = _EPOCH_RE.match(n)
        if m and os.path.isdir(os.path.join(root, n)):
            out.append(int(m.group(1)))
    return sorted(out)


def read_manifest(root: str, k: int) -> typing.Optional[dict]:
    """Epoch ``k``'s ``DONE.json`` when the epoch is COMPLETE, else ``None``.

    Complete = the manifest parses, names its epoch, and the npz it names
    exists. Anything less (no manifest: a build still running or one that
    crashed; an unreadable one; a missing npz) is not an epoch. Also the
    epoch reader of :class:`~.para_search.ParaSearchHolder`.

    Returns:
        dict or None: The manifest plus ``npz_path`` (absolute path of its
        ``npz``, default :data:`REFEREED_NPZ`) and ``npz_filtered_path``
        (the filtered npz, or ``None`` when absent).
    """
    d = epoch_dir(root, k)
    try:
        with open(os.path.join(d, MANIFEST)) as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(man, dict) or int(man.get("epoch", -1)) != int(k):
        return None
    npz = os.path.join(d, str(man.get("npz", REFEREED_NPZ)))
    if not os.path.exists(npz):
        return None
    man["npz_path"] = npz
    # the FILTERED container (head builds since 2026-10-15); optional: an
    # epoch without it serves the full refit where the filtered is chosen
    filt = man.get("npz_filtered")
    filt_path = os.path.join(d, str(filt)) if filt else None
    man["npz_filtered_path"] = (filt_path if filt_path is not None
                                and os.path.exists(filt_path) else None)
    return man


def latest_complete_epoch(root: str, newer_than: typing.Optional[int] = None):
    """``(k, manifest)`` of the newest COMPLETE epoch, or ``None``.

    Scans newest first and stops at the first complete one (incomplete
    epochs are skipped, never waited on). ``newer_than`` restricts the scan
    to ``k > newer_than`` -- the move's per-propose poll, which then costs a
    directory listing and nothing else when there is nothing new.
    """
    for k in reversed(list_epochs(root)):
        if newer_than is not None and k <= int(newer_than):
            return None
        man = read_manifest(root, k)
        if man is not None:
            return k, man
    return None


def next_epoch_index(root: str) -> int:
    """``1 + max(every epoch dir)``: a crashed build's directory is never reused."""
    ks = list_epochs(root)
    return (ks[-1] + 1) if ks else 0


def prune_epochs(root: str, keep_last: int = KEEP_LAST_EPOCHS,
                 keep_every: int = KEEP_EVERY_EPOCH) -> typing.Tuple[int, int]:
    """Strip old epochs down to their ``DONE.json``: ``(n_epochs, bytes_freed)``.

    Kept whole: the newest ``keep_last`` COMPLETE epochs and every epoch
    whose index is a multiple of ``keep_every`` (a long-run record to compare
    refits over time; ``<= 0`` keeps none extra). Every other epoch OLDER
    than the newest complete one loses every file but its manifest, so the
    build history (rows, components, walls) stays readable while the epoch
    stops being complete (its npz is gone) and no move adopts it again. A
    crashed build's leftovers go the same way. Directories are never removed
    (:func:`next_epoch_index` must not reuse an index) and nothing at or
    above the newest complete epoch is touched (a build in progress). Safe
    while the moves run: a rank reads an epoch's npz into memory when it
    adopts it, and the head directs the ranks to the newest one -- or, on a
    build propose under the adoption lag, to the one before it, which the
    ranks load AFTER this prune (the head builder then keeps at least the
    newest two whole).
    Each epoch keeps ~40-70 MB of npz files; unpruned, a refit every stored
    row grows a run directory by ~10 GB a day.
    """
    ks = list_epochs(root)
    names = {}
    for k in ks:
        try:
            names[k] = os.listdir(epoch_dir(root, k))
        except OSError:
            names[k] = []
    # A stripped epoch holds its manifest only: never complete, so skip its
    # JSON read (one per epoch per build adds up when gb_search_1 refits
    # every row on a shared filesystem).
    complete = [k for k in ks if any(n != MANIFEST for n in names[k])
                and read_manifest(root, k) is not None]
    if not complete:
        return 0, 0
    newest = complete[-1]
    keep = set(complete[-max(1, int(keep_last)):])
    n_pruned = n_bytes = 0
    for k in ks:
        if k >= newest or k in keep or (int(keep_every) > 0
                                        and k % int(keep_every) == 0):
            continue
        d = epoch_dir(root, k)
        removed = False
        for name in names[k]:
            p = os.path.join(d, name)
            if name == MANIFEST or not os.path.isfile(p):
                continue
            try:
                size = os.path.getsize(p)
                os.remove(p)
            except OSError:
                continue
            n_bytes += size
            removed = True
        n_pruned += int(removed)
    return n_pruned, n_bytes


def _write_manifest(d: str, manifest: dict) -> None:
    """``DONE.json`` atomically: tmp + fsync + ``os.replace`` (the completion marker)."""
    tmp = os.path.join(d, MANIFEST + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, os.path.join(d, MANIFEST))


# ---------------------------------------------------------------------------
# the stage clock
# ---------------------------------------------------------------------------
_SEARCH_STAGE_RE = re.compile(r"^gb_search_(\d+)$")


def warm_self_stage_kind(stage: typing.Optional[str]) -> typing.Optional[str]:
    """``"search"`` for ``gb_search_N`` with N >= 1, ``"pe"`` for ``full_pe``, else None.

    Every NUMBERED search stage takes the search cadence (user ruling
    2026-10-14: "gb search 1 and gb search 2 will have warm start refits
    every iteration"; until then only N >= 2 built), ``full_pe`` the PE
    cadence. Every other stage (``gb_search_seed``, ``source_search``,
    ``noise_search``, ``replica_pe``) never builds: the seed stage has no
    rows of its own worth refitting and the parent mixture carries it.
    """
    if not stage:
        return None
    s = str(stage)
    if s == "full_pe":
        return "pe"
    m = _SEARCH_STAGE_RE.match(s)
    if m and int(m.group(1)) >= 1:
        return "search"
    return None


def warm_self_cadence(stage, every_search: int, every_pe: int) -> typing.Optional[int]:
    """Stored rows between builds in ``stage``; ``None`` = never builds there."""
    kind = warm_self_stage_kind(stage)
    every = {"search": every_search, "pe": every_pe}.get(kind)
    if every is None or int(every) <= 0:
        return None
    return int(every)


def warm_self_due(rows: int, stage, stage_start: typing.Optional[int],
                  last_built_at: typing.Optional[int], every_search: int,
                  every_pe: int):
    """``(due, cadence, clock)`` after a save that left ``rows`` stored rows.

    ``clock = max(stage_start, last_built_at)``: the stage's own start row
    (its recipe ``start_iteration``, or the previous step's
    ``completed_iteration``) or the latest complete self epoch's
    ``built_at_row``, whichever is later. A build is due once
    ``rows - clock >= cadence``. Both inputs are read from DISK at every
    save, so a resume picks the clock up exactly where it was: it never
    rebuilds on its first save unless the cadence has really elapsed. An
    epoch built AFTER ``rows`` (a rewound store) is ignored for the clock.

    Args:
        rows: Stored rows so far.
        stage: The active stage name.
        stage_start: Its start row (``None`` = 0).
        last_built_at: The latest complete epoch's ``built_at_row`` (``None``
            = no epoch).
        every_search: ``GB_WARM_SELF_EVERY_SEARCH``.
        every_pe: ``GB_WARM_SELF_EVERY_PE``.

    Returns:
        tuple: ``(due (bool), cadence (int or None), clock (int))``;
        ``cadence`` is ``None`` (and ``due`` False) in a stage that never
        builds.
    """
    cadence = warm_self_cadence(stage, every_search, every_pe)
    start = int(stage_start or 0)
    clock = start
    if last_built_at is not None and int(last_built_at) <= int(rows):
        clock = max(start, int(last_built_at))
    if cadence is None:
        return False, None, clock
    return (int(rows) - clock) >= cadence, cadence, clock


def store_stage(g) -> typing.Tuple[typing.Optional[str], int]:
    """``(active stage name, its start row)`` from the store's recipe group.

    ``g`` is the open run group (``f["global_fit"]``). The active stage is
    the first step, in ``order num`` order, whose ``status`` is False; its
    start is its ``start_iteration`` stamp (written at the previous stage's
    boundary by ``completed_recipe_step``), else the previous step's
    ``completed_iteration``, else 0. ``(None, 0)`` without a recipe or when
    every step is complete.
    """
    rg = g.get("recipe") if hasattr(g, "get") else None
    if rg is None:
        return None, 0
    steps = []
    for name in rg.keys():
        a = rg[name].attrs
        try:
            order = int(a["order num"])
        except (KeyError, TypeError, ValueError):
            continue
        steps.append((order, str(name), bool(a.get("status", False)),
                      a.get("start_iteration"), a.get("completed_iteration")))
    steps.sort()
    prev_done = None
    for order, name, status, start, done in steps:
        if not status:
            if start is not None:
                return name, int(start)
            return name, int(prev_done) if prev_done is not None else 0
        prev_done = done
    return None, 0


# ---------------------------------------------------------------------------
# the store read on the saver
# ---------------------------------------------------------------------------
def copy_leaf_rows(g, out_path: str, last_k: int, rows: int,
                   branch: str = "gb") -> dict:
    """Copy the last ``last_k`` stored COLD rows of ``branch`` into a small store.

    ``g`` is the run group of the OPEN source file (the saver's own
    backend handle); ``rows`` its ``attrs["iteration"]`` (rows past it are
    stale). Writes ``out_path`` in the same layout the warm-start pipeline
    reads (``global_fit/chain/<br>``, ``inds/<br>``, ``log_like``, attr
    ``iteration``), with each (row, walker)'s ALIVE leaves packed into the
    first slots and the leaf axis trimmed to the largest alive count: the
    fitter and the referee read alive rows only (``np.nonzero(inds)``), so
    slot identity carries nothing they use. One row is read at a time.
    Used by the saver build (:class:`WarmSelfHook`) only; the head build
    writes the same file from its buffer (:meth:`WarmSelfBuffer.write`).

    Args:
        g: The open source run group.
        out_path: The small store to write (via a ``.tmp`` name).
        last_k: STORED ROWS to copy, counted back from ``rows``.
        rows: The source's ``attrs["iteration"]``.
        branch: The branch. Default ``"gb"``.

    Returns:
        dict: Provenance: ``rows`` (``[a, b]`` inclusive), ``n_rows``,
        ``n_walkers``, ``n_samples``, ``n_leaf_rows``, ``max_alive``.
    """
    a, packed, lls, ntail = _read_packed_rows(g, last_k, rows, branch)
    return _write_packed_rows(out_path, packed, lls, a, ntail, branch)


def _read_packed_rows(g, last_k: int, rows: int, branch: str = "gb"):
    """The last ``last_k`` stored COLD rows of ``branch``, one row at a time.

    Returns ``(first_row, packed, log_likes, n_tail_axes)``: ``packed`` is a
    list (rows) of lists (walkers) of the walker's ALIVE leaf coordinates
    ``(n_alive, ndim)``; ``log_likes`` a list of ``(nw,)`` cold rows;
    ``n_tail_axes`` the store's axes between the row and walker axes
    (``(nsamplers, ntemps)`` -> 2).
    """
    chain = g["chain"][branch]
    inds = g["inds"][branch]
    ll = g["log_like"]
    rows = int(rows)
    if rows <= 0:
        raise ValueError("the store holds no rows yet")
    a = max(0, rows - int(last_k))
    tail = tuple(chain.shape[1:-3])            # (nsamplers, ntemps) or (ntemps,)
    cold = (0,) * len(tail)
    nw = int(chain.shape[-3])
    packed = []
    for it in range(a, rows):
        c_it = np.asarray(chain[(it,) + cold])         # (nw, nleaves, ndim)
        i_it = np.asarray(inds[(it,) + cold], dtype=bool)
        packed.append([np.asarray(c_it[w, np.flatnonzero(i_it[w])],
                                  dtype=np.float64) for w in range(nw)])
    llv = np.asarray(ll[a:rows])[(slice(None),) + cold + (slice(None),)]
    lls = [np.asarray(r, dtype=np.float64).reshape(nw) for r in llv]
    return a, packed, lls, len(tail)


def _write_packed_rows(out_path: str, packed, lls, first_row: int,
                       ntail: int = 2, branch: str = "gb") -> dict:
    """Write rows of per-walker alive leaves in the layout the pipeline reads.

    ``global_fit/chain/<br>`` ``(K, 1.., nw, M, ndim)`` with each (row,
    walker)'s alive leaves packed into the first slots and the leaf axis
    trimmed to the largest alive count, ``inds/<br>``, ``log_like``
    ``(K, 1.., nw)``, attrs ``iteration = K`` and ``source_rows``. Written to
    a temporary name and moved into place. Returns the provenance.
    """
    import h5py

    K = len(packed)
    if K == 0:
        raise ValueError("no rows to write")
    nw = len(packed[0])
    ndim = next((int(x.shape[-1]) for row in packed for x in row
                 if np.ndim(x) == 2 and x.shape[-1]), 9)
    M = max([1] + [len(x) for row in packed for x in row])
    ones = (1,) * int(ntail)
    cold = (0,) * int(ntail)
    ch = np.zeros((K,) + ones + (nw, M, ndim), dtype=np.float64)
    ind = np.zeros((K,) + ones + (nw, M), dtype=bool)
    n_leaf_rows = 0
    for r, per_w in enumerate(packed):
        for w, x in enumerate(per_w):
            n = len(x)
            if n:
                ch[(r,) + cold + (w, slice(0, n))] = x
                ind[(r,) + cold + (w, slice(0, n))] = True
            n_leaf_rows += n
    llv = np.stack([np.asarray(v, dtype=np.float64) for v in lls]).reshape(
        (K,) + ones + (nw,))
    last = int(first_row) + K - 1
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        gg = f.create_group("global_fit")
        gg.attrs["iteration"] = K
        gg.attrs["source_rows"] = [int(first_row), last]
        gg.create_group("chain").create_dataset(branch, data=ch,
                                                compression="gzip")
        gg.create_group("inds").create_dataset(branch, data=ind,
                                               compression="gzip")
        gg.create_dataset("log_like", data=llv)
    os.replace(tmp, out_path)
    return dict(rows=[int(first_row), last], n_rows=K, n_walkers=nw,
                n_samples=K * nw, n_leaf_rows=int(n_leaf_rows),
                max_alive=int(M))


def _to_host(a):
    """cupy-or-numpy -> numpy."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


class WarmSelfBuffer:
    """The HEAD's copy of the last ``last_k`` stored cold GB rows (all walkers).

    The head cannot read the store while the run is going (the saver is the
    only writer and its writes are asynchronous), so it keeps its own copy:
    :meth:`fill_from_store` at start-up (before the first save, so the read
    races nothing) and :meth:`append_state` with every state it hands to the
    saver. :meth:`write` produces exactly the file :func:`copy_leaf_rows`
    makes from the store. ``stage_name`` / ``stage_start`` are the store's
    active stage at the fill (the head builder's resume clock). Built by
    :func:`make_head_builder` and attached as the head backend's
    ``warm_self_buffer``.

    Args:
        last_k (int): Rows held (``GB_WARM_SELF_LAST_K``; a deque of that
            length, one entry per STORED ROW, every walker).
        branch (str, optional): Default ``"gb"``.
        ntail (int, optional): Axes between the row and walker axes in the
            written layout (2 = ``(nsamplers, ntemps)``). Default ``2``;
            :meth:`fill_from_store` takes the store's.

    Attributes:
        n_stored (int): Rows the store holds (at the fill) plus every row
            appended since: the head's row clock.
        stage_name (str or None): The store's active stage at the fill.
        stage_start (int): Its start row.

    Raises:
        ValueError: When ``last_k < 1``.
    """

    def __init__(self, last_k: int, branch: str = "gb", ntail: int = 2):
        import collections

        self.last_k = int(last_k)
        if self.last_k < 1:
            raise ValueError(f"last_k must be >= 1, got {last_k}")
        self.branch = str(branch)
        self.ntail = int(ntail)
        self._rows = collections.deque(maxlen=self.last_k)
        self.n_stored = 0
        self.stage_name: typing.Optional[str] = None
        self.stage_start = 0

    def __len__(self) -> int:
        return len(self._rows)

    def fill_from_store(self, g) -> None:
        """Load the last ``last_k`` rows of the open run group ``g``."""
        rows = int(g.attrs.get("iteration", 0))
        self._rows.clear()
        self.n_stored = rows
        self.stage_name, self.stage_start = store_stage(g)
        if rows <= 0 or "chain" not in g or self.branch not in g["chain"]:
            return
        a, packed, lls, ntail = _read_packed_rows(g, self.last_k, rows,
                                                  self.branch)
        self.ntail = ntail
        for k, (per_w, ll) in enumerate(zip(packed, lls)):
            self._rows.append((per_w, ll, a + k))

    def append_state(self, state) -> None:
        """Append the COLD rung of ``state`` (the state the head saves); ``n_stored`` += 1.

        Called by the head's ``GFHDFBackend.save_step`` for every state it
        hands to the saver.
        """
        br = state.branches[self.branch]
        coords = _to_host(br.coords)[0]
        inds = _to_host(br.inds)[0].astype(bool)
        per_w = [np.array(coords[w, inds[w]], dtype=np.float64)
                 for w in range(coords.shape[0])]
        ll = np.array(_to_host(state.log_like)[0], dtype=np.float64)
        self._rows.append((per_w, ll, self.n_stored))
        self.n_stored += 1

    def write(self, out_path: str) -> dict:
        """The held rows in :func:`copy_leaf_rows`' layout; its provenance."""
        if not self._rows:
            raise ValueError("the head's warm-self buffer holds no rows")
        rows = list(self._rows)
        return _write_packed_rows(out_path, [r[0] for r in rows],
                                  [r[1] for r in rows], rows[0][2],
                                  self.ntail, self.branch)


# ---------------------------------------------------------------------------
# the build (fit -> referee -> apply, exactly as build.py runs it)
# ---------------------------------------------------------------------------
def _npz_counts(path: str) -> typing.Tuple[int, int]:
    """``(n_clusters, n_components)`` of a refereed npz."""
    with np.load(path, allow_pickle=False) as d:
        n_clusters = int(len(d["p"]))
        if "gmm_ncomp" in d:
            n_comp = int(np.asarray(d["gmm_ncomp"]).sum())
        else:
            n_comp = int(len(d["means"]))
    return n_clusters, n_comp


def build_self_epoch(leaf_store: str, out_dir: str, *, tobs: float,
                     last_k: int, device: typing.Optional[int] = None,
                     runner=None) -> dict:
    """Run fit -> referee -> apply on ``leaf_store`` into ``out_dir``.

    The SAME three calls :func:`.build.ensure_warm_start_components` makes,
    through the same in-process runner (:func:`.build._default_runner`):
    ``fit_from_store --store S --last-k K --tobs T --out F`` (plus
    ``--gmm-gpu``: the builder's device -- the head's, or the saver's under
    ``GB_WARM_SELF_AT_ENTRY=0`` -- ``cpu`` without one; the parent
    build's ``auto`` would take device 0 of whatever the process sees),
    ``match_referee --npz F --store S`` and ``referee_apply --fit F
    --referee R --out P``; every other option at its CLI default, i.e. the
    production warm start's settings. ``tobs`` is the RUN's own Tobs.
    Returns the walls and counts; raises on any failure (the hook catches).
    The whole pipeline runs inside ``device``'s cupy device context.

    Args:
        leaf_store: The epoch's ``leaf_rows.h5``.
        out_dir: The epoch directory (``self_fit.npz``,
            ``self_fit_referee.npz`` and ``self_refereed.npz`` land there).
        tobs: The run's Tobs [s].
        last_k: Rows in ``leaf_store`` (all of them).
        device: CUDA device for the mixture fit; ``None`` (or no cupy) =
            CPU.
        runner (callable, optional): The step runner. Default
            :func:`.build._default_runner`.

    Returns:
        dict: ``npz`` (:data:`REFEREED_NPZ`), ``n_clusters``,
        ``n_components`` and ``walls`` (``fit`` / ``referee`` / ``apply``
        seconds).

    Raises:
        RuntimeError: When a step fails or writes no output.
    """
    from .build import _default_runner, _require

    runner = runner or _default_runner
    fit_npz = os.path.join(out_dir, FIT_NPZ)
    referee_npz = os.path.splitext(fit_npz)[0] + "_referee.npz"
    out_npz = os.path.join(out_dir, REFEREED_NPZ)
    walls = {}
    xp = None
    if device is not None:
        try:
            import cupy as xp  # noqa: F401
        except Exception:      # noqa: BLE001 -- a CPU node
            xp = None
    from ...utils.device import device_context

    with device_context(xp, device if xp is not None else None):
        t0 = time.perf_counter()
        runner(["fit_from_store", "--store", leaf_store, "--last-k",
                str(int(last_k)), "--tobs", repr(float(tobs)), "--out",
                fit_npz, "--gmm-gpu",
                "cpu" if (device is None or xp is None) else str(int(device))])
        _require(fit_npz, "fit_from_store")
        walls["fit"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        runner(["match_referee", "--npz", fit_npz, "--store", leaf_store])
        _require(referee_npz, "match_referee")
        walls["referee"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        runner(["referee_apply", "--fit", fit_npz, "--referee", referee_npz,
                "--out", out_npz])
        _require(out_npz, "referee_apply")
        walls["apply"] = time.perf_counter() - t0
    n_clusters, n_comp = _npz_counts(out_npz)
    return dict(npz=REFEREED_NPZ, n_clusters=n_clusters,
                n_components=n_comp, walls=walls)


# ---------------------------------------------------------------------------
# the FILTERED container: the refit minus what every walker already holds
# ---------------------------------------------------------------------------
def _subset_clusters(data: dict, keep: np.ndarray, ncomp: np.ndarray) -> dict:
    """The refereed npz's arrays restricted to the clusters in ``keep``.

    Per-cluster arrays (first axis = clusters) and per-component GMM arrays
    (``gmm_*`` but ``gmm_ncomp``, first axis = components) are masked;
    ``meta`` and the per-island ``f0_window_edges`` are kept whole (the
    surviving clusters still index their own islands). Any other layout is
    refused rather than guessed.
    """
    C = len(keep)
    comp_keep = np.repeat(keep, ncomp)
    n_comp = int(np.sum(ncomp))
    out = {}
    for k, a in data.items():
        a = np.asarray(a)
        if k in ("meta", "f0_window_edges") or a.ndim == 0:
            out[k] = a
        elif k.startswith("gmm_") and k != "gmm_ncomp" and a.shape[0] == n_comp:
            out[k] = a[comp_keep]
        elif a.shape[0] == C:
            out[k] = a[keep]
        else:
            raise ValueError(
                f"consensus filter: npz key {k!r} with shape {a.shape} is "
                f"neither per cluster ({C}) nor per component ({n_comp})")
    return out


def consensus_filter(refereed_npz: str, leaf_store: str, out_npz: str, *,
                     threshold: float = 0.95) -> dict:
    """Write the FILTERED refit: ``refereed_npz`` minus the consensus clusters.

    User ruling 2026-10-15: "new refit removal of sources that are already
    in all walkers", "sources that do not exist in all four (with high
    confidence)". Every leaf row of ``leaf_store`` (the rows the refit was
    fitted from) is assigned to its nearest component with the referee's
    own member re-extraction (:func:`.match_referee.assign_members`); per
    cluster and sample (row x walker) the leaves are counted. A cluster is
    CONSENSUS -- every walker holds it, with high confidence -- when EVERY
    walker holds it in >= ``threshold`` of its rows AND >= ``threshold`` of
    all samples hold the same number of its leaves (>= 1). A blend one
    walker resolves into two sources and another does not is therefore
    kept, as is anything a walker lacks in more than ``1 - threshold`` of
    its rows. Writes the kept clusters in the same schema (so
    :meth:`.proposal.WarmStartComponents.from_npz` loads it unchanged);
    writes nothing when no cluster survives. Returns the counts and the
    ``consensus`` mask.

    Args:
        refereed_npz: The epoch's refereed npz.
        leaf_store: The ``leaf_rows.h5`` it was fitted from (all rows read).
        out_npz: The filtered npz to write (:data:`FILTERED_NPZ`).
        threshold: ``GB_WARM_SELF_CONSENSUS``. Default ``0.95``.

    Returns:
        dict: ``n_clusters``, ``n_kept``, ``n_components_kept``, the
        ``(n_clusters,)`` bool ``consensus`` mask and ``npz`` (the written
        file's basename, or ``None`` when nothing survived).
    """
    from . import match_referee as mr
    from .fit_from_store import load_leaf_table, to_observable

    thr = float(threshold)
    with np.load(refereed_npz, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data["meta"]))
    p = np.asarray(data["p"])
    C = len(p)
    if str(meta.get("basis", "sampling")) == "observable":
        from . import basis as wb

        obs_map = wb.build_map_from_stored_params(meta["map_params"])
        ncomp = np.asarray(data["gmm_ncomp"]).astype(int)
        means = np.asarray(data["gmm_means"])
        covs = np.asarray(data["gmm_covs"])
    else:
        obs_map = None
        ncomp = np.ones(C, dtype=int)
        means, covs = np.asarray(data["means"]), np.asarray(data["covs"])
    comp_cluster = np.repeat(np.arange(C), ncomp)
    island = np.repeat(np.asarray(data["island_id"]), ncomp)
    X, sid, info = load_leaf_table(leaf_store, None)
    nw = int(info["nwalkers"])
    G = to_observable(X, obs_map) if obs_map is not None else X
    assign, _ = mr.assign_members(G, means, covs, island,
                                  np.asarray(data["f0_window_edges"]))
    ok = assign >= 0
    rows = np.unique(sid // nw)
    R = len(rows)
    samp = np.searchsorted(rows, sid // nw) * nw + (sid % nw)
    counts = np.zeros((C, R * nw), dtype=np.int32)
    np.add.at(counts, (comp_cluster[assign[ok]], samp[ok]), 1)
    presence = (counts > 0).reshape(C, R, nw).mean(axis=1)        # (C, nw)
    freq = np.stack([(counts == v).sum(axis=1)
                     for v in range(int(counts.max()) + 1)], axis=1)
    mode = freq.argmax(axis=1)
    mode_frac = freq[np.arange(C), mode] / float(R * nw)
    consensus = ((presence.min(axis=1) >= thr) & (mode_frac >= thr)
                 & (mode >= 1))
    keep = ~consensus
    res = dict(n_clusters=int(C), n_kept=int(keep.sum()),
               n_components_kept=int(ncomp[keep].sum()),
               consensus=consensus, npz=None)
    if keep.any():
        out = _subset_clusters(data, keep, ncomp)
        meta["consensus_filter"] = dict(
            threshold=thr, n_clusters=int(C), n_kept=int(keep.sum()),
            n_rows=int(R), n_walkers=nw, source=os.path.basename(refereed_npz))
        out["meta"] = np.asarray(json.dumps(meta))
        tmp = out_npz + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, out_npz)
        res["npz"] = os.path.basename(out_npz)
    return res


class WarmSelfWatchdog:
    """The monitor hook's watchdog pattern, for the self-refit build.

    Same three signals as :class:`..monitor.hooks.MonitorWatchdog` (a save
    queued when the build finished; the build ate more than
    ``GF_MONITOR_WARN_FRAC`` of the measured save interval; states dropped
    across the build), reported under ``[GB_WARM_SELF]``. Dropped states
    switch the BUILDER off for the rest of the run (the moves keep the
    epochs they hold): the refit is never worth stored iterations. Saver
    build only (``GB_WARM_SELF_AT_ENTRY=0``).

    Attributes:
        disabled_reason (str or None): Set once a build cost stored states.
        builds (int): Builds reported.
        delayed (int): Builds that held a save up.
        total_build_s (float): Summed build walls [s].
    """

    def __init__(self):
        from ..monitor.hooks import MonitorWatchdog

        self._inner = MonitorWatchdog()           # cadence bookkeeping only
        self.disabled_reason: typing.Optional[str] = None
        self.builds = 0
        self.delayed = 0
        self.total_build_s = 0.0

    @property
    def save_interval(self):
        """The measured (running-averaged) save interval [s]; ``None`` before two saves."""
        return self._inner.save_interval

    def note_save(self) -> None:
        """Record one save (the interval clock)."""
        self._inner.note_save()

    def report(self, elapsed: float, *, queued_after: bool,
               dropped_delta: int) -> None:
        """Log a build attempt; ``dropped_delta > 0`` disables the builder for the run.

        Args:
            elapsed: The build wall [s].
            queued_after: A save was waiting when the build finished.
            dropped_delta: Save states dropped across the build.
        """
        from ..monitor.hooks import _num

        iv = self.save_interval
        frac = (elapsed / iv) if iv else None
        warn_frac = _num("GF_MONITOR_WARN_FRAC", 0.25)
        if dropped_delta <= 0:            # a drop verdict is not a new build
            self.builds += 1
            self.total_build_s += elapsed
        if dropped_delta > 0:
            self.disabled_reason = (
                f"dropped {dropped_delta} save state(s) during a build")
            logger.warning(
                "\n########################################################\n"
                "##  [GB_WARM_SELF] STOPPING THE SELF REFIT: THE BUILD    \n"
                "##  COST STORED ITERATIONS.                             \n"
                "##  %d save state(s) were dropped while a build held    \n"
                "##  the saver rank (%.1f s). Those iterations are NOT   \n"
                "##  in the store. No more epochs are built this run;    \n"
                "##  the moves keep the epoch they hold.                 \n"
                "##  Raise GB_WARM_SELF_EVERY_SEARCH / _PE to build less \n"
                "##  often, then relaunch.                               \n"
                "########################################################",
                dropped_delta, elapsed)
            return
        if queued_after or (frac is not None and frac > warn_frac):
            self.delayed += 1
            logger.warning(
                "\n########################################################\n"
                "##  [GB_WARM_SELF] THE SELF-REFIT BUILD IS HOLDING THE   \n"
                "##  SAVER RANK UP.                                       \n"
                "##  build %.1f s%s                                       \n"
                "##  a save was %swaiting when it finished                \n"
                "##  %d of %d builds have delayed a save                  \n"
                "##  The sampler's save_step is a BLOCKING send to this   \n"
                "##  rank: this time shows as a slow iteration.           \n"
                "########################################################",
                elapsed,
                (" = %.0f%% of the %.0f s save interval" % (100 * frac, iv))
                if frac else "",
                "" if queued_after else "NOT ", self.delayed, self.builds)
        else:
            logger.info(
                "[GB_WARM_SELF] build held no save (%.1f s%s; %d built, %d "
                "delayed so far).", elapsed,
                (" = %.0f%% of the %.0f s save interval" % (100 * frac, iv))
                if frac else "", self.builds, self.delayed)


class WarmSelfHook:
    """The saver-rank trigger: after a save, build a self epoch when due.

    Called by ``save_to_backend_asynchronously_and_plot`` once per batch
    that saved a row (after the plots and the monitor notice), with the
    saver's own backend. Never raises into the saver loop. The 2026-10-13
    design, kept behind ``GB_WARM_SELF_AT_ENTRY=0`` (the 9mo / 12mo
    launchers build on the head, so :func:`make_warm_self_hook` returns
    ``None`` there). A queued save always wins; a build reads the last
    ``last_k`` rows through one scoped ``open("r")`` of the saver's backend
    (:func:`copy_leaf_rows`), runs :func:`build_self_epoch` (no consensus
    filter on this path), writes the manifest LAST and prunes.

    Args:
        epoch_root (str): :func:`self_epoch_root` of the store.
        tobs (float): The run's Tobs [s].
        last_k (int): Rows copied per build (``GB_WARM_SELF_LAST_K``).
        every_search (int): ``GB_WARM_SELF_EVERY_SEARCH``.
        every_pe (int): ``GB_WARM_SELF_EVERY_PE``.
        device (int, optional): The saver's GPU for the mixture fit.
        branch (str, optional): Default ``"gb"``.
        builder (callable, optional): Default :func:`build_self_epoch`.
        runner (callable, optional): Forwarded to the builder.
        keep_last (int, optional): Pruning; default ``KEEP_LAST_EPOCHS`` (3).
        keep_every (int, optional): Pruning; default ``KEEP_EVERY_EPOCH``
            (10).

    Attributes:
        watchdog (WarmSelfWatchdog): The build watchdog.
    """

    def __init__(self, *, epoch_root: str, tobs: float, last_k: int,
                 every_search: int, every_pe: int,
                 device: typing.Optional[int] = None, branch: str = "gb",
                 builder=None, runner=None,
                 keep_last: int = KEEP_LAST_EPOCHS,
                 keep_every: int = KEEP_EVERY_EPOCH):
        self.epoch_root = str(epoch_root)
        self.keep_last = int(keep_last)
        self.keep_every = int(keep_every)
        self.tobs = float(tobs)
        self.last_k = int(last_k)
        self.every_search = int(every_search)
        self.every_pe = int(every_pe)
        self.device = None if device is None else int(device)
        self.branch = str(branch)
        self._builder = builder or build_self_epoch
        self._runner = runner
        self.watchdog = WarmSelfWatchdog()
        #: rows at the last FAILED build; retried only after another cadence
        self._failed_at: typing.Optional[int] = None
        self._announced = False
        #: the saver's dropped-state count when the last build ended (None =
        #: no build since the last check), and that build's wall
        self._drop_base: typing.Optional[int] = None
        self._last_elapsed = 0.0

    def note_save(self) -> None:
        """Record one save on the watchdog's interval clock."""
        self.watchdog.note_save()

    def __call__(self, gb_reader, comm, main_rank, i, *,
                 dropped_so_far: int = 0) -> None:
        """Build an epoch if one is due after this save; never raises.

        Args:
            gb_reader: The saver's backend (``open``, ``filename``, ``name``).
            comm: The MPI communicator (``iprobe`` for a queued save), or
                ``None``.
            main_rank (int): The head rank.
            i (int): The save index (unused by the decision).
            dropped_so_far: The saver's dropped-state count.
        """
        try:
            self._maybe_build(gb_reader, comm, main_rank, dropped_so_far)
        except Exception as exc:                  # noqa: BLE001
            logger.warning(
                "[GB_WARM_SELF] after-save hook failed (%s: %s); saves are "
                "unaffected.\n%s", type(exc).__name__, exc,
                traceback.format_exc())

    # -- the decision + the read (one scoped open of the saver's store) ----
    def _read(self, gb_reader):
        """``(rows, stage, stage_start, last_built_at, due, cadence, clock)``."""
        latest = latest_complete_epoch(self.epoch_root)
        last_built = None if latest is None else latest[1].get("built_at_row")
        with gb_reader.open("r") as f:
            g = f[getattr(gb_reader, "name", "global_fit")]
            rows = int(g.attrs["iteration"])
            stage, start = store_stage(g)
        due, cadence, clock = warm_self_due(
            rows, stage, start, last_built, self.every_search, self.every_pe)
        return rows, stage, start, last_built, due, cadence, clock

    def _finish_build(self, t_all, comm, main_rank, dropped_so_far) -> float:
        """Report a build attempt (success or failure) to the watchdog."""
        elapsed = time.perf_counter() - t_all
        self.watchdog.report(
            elapsed,
            queued_after=bool(comm.iprobe(source=main_rank))
            if comm is not None else False,
            dropped_delta=0)
        # States the build cost are dropped while the saver drains the
        # backlog it left, i.e. in the NEXT batch, before the next call.
        self._drop_base = int(dropped_so_far)
        self._last_elapsed = elapsed
        return elapsed

    def _maybe_build(self, gb_reader, comm, main_rank, dropped_so_far):
        """The decision and the build of :meth:`__call__` (may raise; the caller logs)."""
        if self._drop_base is not None:
            delta = int(dropped_so_far) - self._drop_base
            self._drop_base = None
            if delta > 0:
                self.watchdog.report(self._last_elapsed, queued_after=False,
                                     dropped_delta=delta)
        if self.watchdog.disabled_reason is not None:
            return
        # A queued save always wins: the cadence condition stays true, so the
        # build simply happens after a later save.
        if comm is not None and comm.iprobe(source=main_rank):
            return
        rows, stage, start, last_built, due, cadence, clock = self._read(
            gb_reader)
        if not self._announced:
            self._announced = True
            logger.info(
                "[GB_WARM_SELF] builder armed on the saver: stage %s (start "
                "row %d), %d row(s) stored, latest complete epoch built at "
                "%s; cadence search %d / pe %d rows, last %d rows x all "
                "walkers, Tobs %.6g s, device %s, epochs under %s.",
                stage, start, rows, last_built, self.every_search,
                self.every_pe, self.last_k, self.tobs, self.device,
                self.epoch_root)
        if not due:
            return
        if self._failed_at is not None and cadence is not None and (
                rows - self._failed_at) < cadence:
            return
        k = next_epoch_index(self.epoch_root)
        d = epoch_dir(self.epoch_root, k)
        os.makedirs(d, exist_ok=False)
        t_all = time.perf_counter()
        leaf = os.path.join(d, LEAF_ROWS_H5)
        try:
            t0 = time.perf_counter()
            # THE STORE READ: the saver's own backend open, scoped, closed
            # before anything slow runs (and before this hook returns).
            with gb_reader.open("r") as f:
                g = f[getattr(gb_reader, "name", "global_fit")]
                prov = copy_leaf_rows(g, leaf, self.last_k, rows,
                                      branch=self.branch)
            t_read = time.perf_counter() - t0
            kw = dict(tobs=self.tobs, last_k=prov["n_rows"],
                      device=self.device)
            if self._runner is not None:
                kw["runner"] = self._runner
            res = self._builder(leaf, d, **kw)
            walls = dict(res.get("walls", {}))
            walls["read"] = t_read
            walls["total"] = time.perf_counter() - t_all
            manifest = dict(
                epoch=int(k), built_at_row=int(rows), stage=stage,
                stage_start=int(start), clock=int(clock),
                cadence=cadence, rows=prov["rows"], n_rows=prov["n_rows"],
                n_walkers=prov["n_walkers"], n_samples=prov["n_samples"],
                n_leaf_rows=prov["n_leaf_rows"],
                n_clusters=int(res["n_clusters"]),
                n_components=int(res["n_components"]),
                npz=str(res.get("npz", REFEREED_NPZ)), tobs=self.tobs,
                device=self.device,
                source_store=os.path.abspath(str(gb_reader.filename)),
                walls={kk: round(float(v), 3) for kk, v in walls.items()},
                created_unix=time.time(),
            )
            _write_manifest(d, manifest)
        except Exception:
            self._failed_at = rows
            logger.warning(
                "[GB_WARM_SELF] epoch %d build FAILED at row %d (stage %s); "
                "the epoch directory %s stays incomplete and is ignored; "
                "retry after another %s row(s).\n%s", k, rows, stage, d,
                cadence, traceback.format_exc())
            self._finish_build(t_all, comm, main_rank, dropped_so_far)
            return
        finally:
            try:
                os.remove(leaf)
            except OSError:
                pass
        self._failed_at = None
        elapsed = time.perf_counter() - t_all
        a, b = prov["rows"]
        logger.info(
            "[GB_WARM_SELF] epoch %d built from rows %d..%d (%d samples, %d "
            "components) in %.1f s (stage %s, built at row %d; read %.1f s, "
            "fit %.1f s, referee %.1f s, apply %.1f s).", k, a, b,
            prov["n_samples"], int(res["n_components"]), elapsed, stage, rows,
            walls.get("read", 0.0), walls.get("fit", 0.0),
            walls.get("referee", 0.0), walls.get("apply", 0.0))
        self._finish_build(t_all, comm, main_rank, dropped_so_far)
        self._prune()

    def _prune(self) -> None:
        """Strip older epochs after a successful build (:func:`prune_epochs`)."""
        try:
            n, freed = prune_epochs(self.epoch_root, self.keep_last,
                                    self.keep_every)
        except Exception as exc:                  # noqa: BLE001
            logger.warning(
                "[GB_WARM_SELF] pruning older epochs failed (%s: %s); the "
                "build is unaffected.", type(exc).__name__, exc)
            return
        if n:
            logger.info(
                "[GB_WARM_SELF] pruned %d older epoch(s) to their %s (kept "
                "whole: the newest %d and every %d-th); freed %.1f MB.", n,
                MANIFEST, self.keep_last, self.keep_every, freed / 1e6)


def _parent_warm_start(gb) -> str:
    """The run's parent warm-start npz path ('' when it has none)."""
    return str(getattr(gb, "warm_start_components", "")
               or os.environ.get("GB_WARM_START_COMPONENTS", "") or "").strip()


def make_warm_self_hook(curr, *, store_path: typing.Optional[str] = None):
    """The saver's :class:`WarmSelfHook`, or ``None`` when the saver does not build.

    Off unless ``GBSettings.warm_self`` (``GB_WARM_SELF=1``) AND the run has
    a parent warm start (``warm_start_components`` /
    ``GB_WARM_START_COMPONENTS``): the self epochs only feed the warm moves,
    which exist only with one. Also off when ``warm_self_at_entry``
    (``GB_WARM_SELF_AT_ENTRY=1``, the default since 2026-10-15): the HEAD
    builds then (:func:`make_head_builder`); never both. Called on the saver
    rank by ``run.py``; returns ``None`` on the 9mo / 12mo runs
    (``GB_WARM_SELF_AT_ENTRY=1``).

    Args:
        curr: The run's ``GlobalFitSetup`` (``source_info["gb"]``,
            ``general_info``).
        store_path: The store h5. Default ``None`` =
            ``general_info.main_file_path``.

    Returns:
        WarmSelfHook or None: The hook (device = ``general_info.gpus[0]``,
        Tobs = ``general_info.Tobs``), or ``None``.

    Raises:
        ValueError: When ``GB_WARM_SELF_LAST_K < 1``.
    """
    gb = (getattr(curr, "source_info", None) or {}).get("gb")
    if gb is None or not bool(getattr(gb, "warm_self", False)):
        return None
    if bool(getattr(gb, "warm_self_at_entry", False)):
        return None
    if not _parent_warm_start(gb):
        logger.warning(
            "[GB_WARM_SELF] GB_WARM_SELF=1 but the run has no parent warm "
            "start (GB_WARM_START_COMPONENTS unset): there is no warm move to "
            "feed, so no self epochs are built.")
        return None
    gi = curr.general_info
    path = store_path or gi.main_file_path
    gpus = list(getattr(gi, "gpus", None) or [])
    last_k = int(gb.warm_self_last_k)
    if last_k < 1:
        raise ValueError(f"GB_WARM_SELF_LAST_K must be >= 1, got {last_k}")
    return WarmSelfHook(
        epoch_root=self_epoch_root(path), tobs=float(gi.Tobs),
        last_k=last_k,
        every_search=int(gb.warm_self_every_search),
        every_pe=int(gb.warm_self_every_pe),
        device=(int(gpus[0]) if gpus else None))


# ---------------------------------------------------------------------------
# the head-side build at the warm move's entry (user ruling 2026-10-15)
# ---------------------------------------------------------------------------
#: the head builder of each epoch root in THIS process (only the head
#: registers one; a compute rank's mixer finds none and never builds)
_HEAD_BUILDERS: dict = {}


def register_head_builder(builder) -> None:
    """Make ``builder`` the head builder of its ``epoch_root`` in this process."""
    _HEAD_BUILDERS[os.path.abspath(str(builder.epoch_root))] = builder


# TODO check: keep/remove/document -- no caller outside tests (test cleanup only)
def unregister_head_builder(epoch_root: str) -> None:
    """Drop the head builder registered for ``epoch_root`` here (a no-op when none is)."""
    _HEAD_BUILDERS.pop(os.path.abspath(str(epoch_root)), None)


def head_builder_for(epoch_root: str):
    """The head builder registered for ``epoch_root`` here, or ``None``."""
    return _HEAD_BUILDERS.get(os.path.abspath(str(epoch_root)))


class WarmSelfHeadBuilder:
    """Build a self epoch ON THE HEAD when a warm move starts and the cadence is due.

    User ruling 2026-10-15: "Let's just build everything on the head at entry
    time for the refit ... Make that the default." Called by
    :meth:`WarmSelfMixer.begin_propose` on the head with the stage the
    propose runs in (re-stamped per propose by the stage combine). The
    clock is :func:`warm_self_due`'s: a numbered search stage builds once at
    least ``every_search`` rows were stored since the stage start or the
    last build (every search CYCLE at the default 1), ``full_pe`` every
    ``every_pe`` rows (25), nothing else ever builds; and a stage's FIRST
    warm entry with no refit built in it yet builds at once from the rows
    already stored (review 2026-10-15: gb_search_1's first cycle then draws
    from a refit of the seed population instead of the parent mixture). The
    stage start is the store's (the buffer's fill) on a resume, else the
    stored row count when the head first sees the stage at a warm entry; the
    last build is read from disk. Rows are the states the head handed to the
    saver (``row_count: head states``): an epoch numbered past the stored
    rows after a resume (states the saver dropped) is ignored by the clock.
    Only the head rank builds (``begin_propose`` passes its rank). A build
    writes the buffer to ``leaf_rows.h5``, runs the parent pipeline
    (:func:`build_self_epoch`), then :func:`consensus_filter`, then the
    manifest LAST, then prunes; every rank waits for it (the compute ranks
    are parked on the propose). It never raises: a failure logs, retries
    after another cadence, and the propose goes on with the epoch it has.

    ``adopt_lag`` (``GB_WARM_SELF_ADOPT_LAG``, ruling 2026-10-16) only words
    the build line and guards the prune: the MIXER holds the new epoch back
    for one draw (:meth:`WarmSelfMixer.begin_propose`), so the build line
    names the epoch this propose uses instead (the newest complete one
    before the build), and the prune keeps at least the newest two epochs
    whole, because the ranks load that previous one after it.
    ``last_built_epoch`` is the index of the latest successful build.

    Args:
        epoch_root (str): :func:`self_epoch_root` of the store.
        buffer (WarmSelfBuffer): The head's row buffer.
        tobs (float): The run's Tobs [s].
        last_k (int): ``GB_WARM_SELF_LAST_K`` (the buffer's length).
        every_search (int): ``GB_WARM_SELF_EVERY_SEARCH``.
        every_pe (int): ``GB_WARM_SELF_EVERY_PE``.
        device (int, optional): The head's GPU for the mixture fit.
        consensus (float, optional): ``GB_WARM_SELF_CONSENSUS``. Default
            ``0.95``.
        branch (str, optional): Default ``"gb"``.
        builder (callable, optional): Default :func:`build_self_epoch`.
        runner (callable, optional): Forwarded to the builder.
        filter_fn (callable, optional): Default :func:`consensus_filter`.
        keep_last (int, optional): Pruning; default ``KEEP_LAST_EPOCHS`` (3;
            at least 2 under the lag).
        keep_every (int, optional): Pruning; default ``KEEP_EVERY_EPOCH``
            (10).
        adopt_lag (bool, optional): ``GB_WARM_SELF_ADOPT_LAG``. Default
            ``True``.

    Attributes:
        n_built (int): Successful builds in this process.
        last_built_epoch (int or None): Index of the latest successful build.
    """

    def __init__(self, *, epoch_root: str, buffer: WarmSelfBuffer,
                 tobs: float, last_k: int, every_search: int, every_pe: int,
                 device: typing.Optional[int] = None, consensus: float = 0.95,
                 branch: str = "gb", builder=None, runner=None,
                 filter_fn=None, keep_last: int = KEEP_LAST_EPOCHS,
                 keep_every: int = KEEP_EVERY_EPOCH, adopt_lag: bool = True):
        self.epoch_root = str(epoch_root)
        self.buffer = buffer
        self.tobs = float(tobs)
        self.last_k = int(last_k)
        self.every_search = int(every_search)
        self.every_pe = int(every_pe)
        self.device = None if device is None else int(device)
        self.consensus = float(consensus)
        self.branch = str(branch)
        self.keep_last = int(keep_last)
        self.keep_every = int(keep_every)
        self.adopt_lag = bool(adopt_lag)
        self._builder = builder or build_self_epoch
        self._runner = runner
        self._filter = filter_fn or consensus_filter
        self._stage: typing.Optional[str] = None
        self._stage_start = 0
        self._failed_at: typing.Optional[int] = None
        self.n_built = 0
        self.last_built_epoch: typing.Optional[int] = None

    def _stage_clock_start(self, stage: str, rows: int) -> int:
        """The stage's start row: the store's on a resume, else ``rows`` at first sight."""
        if stage != self._stage:
            self._stage = stage
            if stage == self.buffer.stage_name:
                self._stage_start = int(self.buffer.stage_start)
            else:
                self._stage_start = int(rows)
        return self._stage_start

    def maybe_build(self, stage) -> bool:
        """Build an epoch now if one is due in ``stage``; True when one was built."""
        try:
            return self._maybe_build(stage)
        except Exception as exc:                  # noqa: BLE001
            logger.warning(
                "[GB_WARM_SELF] head build check failed (%s: %s); the "
                "propose goes on with the epoch it holds.\n%s",
                type(exc).__name__, exc, traceback.format_exc())
            return False

    def _maybe_build(self, stage) -> bool:
        """The decision and the build of :meth:`maybe_build` (may raise; the caller logs)."""
        if not stage:
            return False
        stage = str(stage)
        rows = int(self.buffer.n_stored)
        first_entry = stage != self._stage
        start = self._stage_clock_start(stage, rows)
        latest = latest_complete_epoch(self.epoch_root)
        last_built = None if latest is None else latest[1].get("built_at_row")
        due, cadence, clock = warm_self_due(
            rows, stage, start, last_built, self.every_search, self.every_pe)
        # A stage's FIRST warm entry with no refit built in it yet builds at
        # once, from the rows already stored (gb_search_1's first cycle then
        # draws from a refit of the seed / source_search population, full_pe's
        # first warm draw from the end of the search). A resume inside a stage
        # that already refitted keeps the clock (last_built >= its start).
        if (not due and first_entry and cadence is not None
                and (last_built is None or int(last_built) < start)):
            due = True
        if not due or len(self.buffer) == 0:
            return False
        if self._failed_at is not None and cadence is not None and (
                rows - self._failed_at) < cadence:
            return False
        k = next_epoch_index(self.epoch_root)
        d = epoch_dir(self.epoch_root, k)
        os.makedirs(d, exist_ok=False)
        leaf = os.path.join(d, LEAF_ROWS_H5)
        t_all = time.perf_counter()
        walls: dict = {}
        try:
            t0 = time.perf_counter()
            prov = self.buffer.write(leaf)
            walls["read"] = time.perf_counter() - t0
            kw = dict(tobs=self.tobs, last_k=prov["n_rows"], device=self.device)
            if self._runner is not None:
                kw["runner"] = self._runner
            res = self._builder(leaf, d, **kw)
            walls.update(dict(res.get("walls", {})))
            npz = str(res.get("npz", REFEREED_NPZ))
            t0 = time.perf_counter()
            filt = self._filter(os.path.join(d, npz), leaf,
                                os.path.join(d, FILTERED_NPZ),
                                threshold=self.consensus)
            walls["filter"] = time.perf_counter() - t0
            walls["total"] = time.perf_counter() - t_all
            has_filtered = os.path.exists(os.path.join(d, FILTERED_NPZ))
            n_clusters = int(res.get("n_clusters", filt.get("n_clusters", 0)))
            n_kept = int(filt.get("n_kept", 0))
            manifest = dict(
                epoch=int(k), built_at_row=int(rows), stage=stage,
                stage_start=int(start), clock=int(clock), cadence=cadence,
                rows=prov["rows"], n_rows=prov["n_rows"],
                n_walkers=prov["n_walkers"], n_samples=prov["n_samples"],
                n_leaf_rows=prov["n_leaf_rows"], n_clusters=n_clusters,
                n_components=int(res["n_components"]), npz=npz,
                npz_filtered=FILTERED_NPZ if has_filtered else None,
                n_clusters_filtered=n_kept if has_filtered else 0,
                n_components_filtered=int(filt.get("n_components_kept", 0)),
                consensus=self.consensus, built_on="head",
                # rows / built_at_row count the states the head handed to the
                # saver; the store holds fewer if the saver ever dropped one
                # while behind (warm_self_due ignores an epoch past the rows)
                row_count="head states", tobs=self.tobs,
                device=self.device,
                walls={kk: round(float(v), 3) for kk, v in walls.items()},
                created_unix=time.time(),
            )
            _write_manifest(d, manifest)
        except Exception:
            self._failed_at = rows
            logger.warning(
                "[GB_WARM_SELF] head epoch %d build FAILED at row %d (stage "
                "%s); the epoch directory %s stays incomplete and is ignored; "
                "retry after another %s row(s).\n%s", k, rows, stage, d,
                cadence, traceback.format_exc())
            return False
        finally:
            try:
                os.remove(leaf)
            except OSError:
                pass
        self._failed_at = None
        self.n_built += 1
        self.last_built_epoch = int(k)
        a, b = prov["rows"]
        # the adoption lag: this propose draws from the newest complete epoch
        # that existed before this build (the mixer polled it first)
        held = ""
        if self.adopt_lag:
            held = ("; held for the next warm draw, this propose uses "
                    + ("the original warm start" if latest is None
                       else f"epoch {int(latest[0])}"))
        logger.info(
            "[GB_WARM_SELF] epoch %d built ON THE HEAD at the %s warm entry "
            "from head states %d..%d (%d samples): %d components; filtered: kept %d "
            "of %d clusters (%d held by every walker dropped) in %.1f s "
            "(read %.1f, fit %.1f, referee %.1f, apply %.1f, filter %.1f); "
            "every rank waited for it%s.", k, stage, a, b, prov["n_samples"],
            int(res["n_components"]), n_kept, n_clusters, n_clusters - n_kept,
            walls["total"], walls.get("read", 0.0), walls.get("fit", 0.0),
            walls.get("referee", 0.0), walls.get("apply", 0.0),
            walls.get("filter", 0.0), held)
        # under the lag the ranks load the epoch BEFORE this one after the
        # prune below: keep at least the newest two whole
        keep_last = max(self.keep_last, 2) if self.adopt_lag else self.keep_last
        try:
            n, freed = prune_epochs(self.epoch_root, keep_last,
                                    self.keep_every)
            if n:
                logger.info(
                    "[GB_WARM_SELF] pruned %d older epoch(s) to their %s "
                    "(kept whole: the newest %d and every %d-th); freed "
                    "%.1f MB.", n, MANIFEST, keep_last, self.keep_every,
                    freed / 1e6)
        except Exception as exc:                  # noqa: BLE001
            logger.warning("[GB_WARM_SELF] pruning older epochs failed "
                           "(%s: %s); the build is unaffected.",
                           type(exc).__name__, exc)
        return True


def make_head_builder(curr, backend=None, *,
                      store_path: typing.Optional[str] = None):
    """The head's :class:`WarmSelfHeadBuilder` (registered), or ``None``.

    On when ``warm_self`` AND ``warm_self_at_entry`` AND the run has a
    parent warm start. Fills the buffer from the store through
    ``backend.open("r")`` NOW -- at the head's set-up, before the first save,
    so the read races no write -- attaches it as ``backend.warm_self_buffer``
    (the head's ``save_step`` appends every state it saves) and registers
    the builder for the run's epoch root. Called once by
    ``GlobalFit.prepare_main`` on the head; armed on the 9mo / 12mo runs.

    Args:
        curr: The run's ``GlobalFitSetup`` (``source_info["gb"]`` with the
            ``warm_self*`` fields, ``general_info``).
        backend (GFHDFBackend, optional): The head's backend. ``None`` = no
            fill and no attachment.
        store_path: The store h5. Default ``None`` =
            ``general_info.main_file_path``.

    Returns:
        WarmSelfHeadBuilder or None: The registered builder (device =
        ``general_info.gpus[0]``, Tobs = ``general_info.Tobs``), or ``None``
        when off.
    """
    gb = (getattr(curr, "source_info", None) or {}).get("gb")
    if gb is None or not bool(getattr(gb, "warm_self", False)):
        return None
    if not bool(getattr(gb, "warm_self_at_entry", False)):
        return None
    if not _parent_warm_start(gb):
        logger.warning(
            "[GB_WARM_SELF] GB_WARM_SELF=1 but the run has no parent warm "
            "start (GB_WARM_START_COMPONENTS unset): there is no warm move to "
            "feed, so the head builds no self epochs.")
        return None
    gi = curr.general_info
    path = store_path or gi.main_file_path
    last_k = int(gb.warm_self_last_k)
    buf = WarmSelfBuffer(last_k=last_k)
    if backend is not None and os.path.exists(str(path)):
        try:
            with backend.open("r") as f:
                name = getattr(backend, "name", "global_fit")
                if name in f:
                    buf.fill_from_store(f[name])
        except Exception as exc:                  # noqa: BLE001
            logger.warning(
                "[GB_WARM_SELF] the head could not read the last %d rows from "
                "%s (%s: %s); its buffer starts empty and fills as it saves.",
                last_k, path, type(exc).__name__, exc)
    if backend is not None:
        backend.warm_self_buffer = buf
    gpus = list(getattr(gi, "gpus", None) or [])
    builder = WarmSelfHeadBuilder(
        epoch_root=self_epoch_root(path), buffer=buf, tobs=float(gi.Tobs),
        last_k=last_k, every_search=int(gb.warm_self_every_search),
        every_pe=int(gb.warm_self_every_pe),
        device=(int(gpus[0]) if gpus else None),
        consensus=float(getattr(gb, "warm_self_consensus", 0.95)),
        adopt_lag=bool(getattr(gb, "warm_self_adopt_lag", True)))
    register_head_builder(builder)
    logger.info(
        "[GB_WARM_SELF] head builder armed: the buffer holds %d of the last %d "
        "stored rows (store rows %d, stage %s from row %d); builds at the warm "
        "move's entry every %d search / %d PE rows, consensus %.2f, device "
        "%s, epochs under %s; adoption lag %s.", len(buf), last_k,
        buf.n_stored, buf.stage_name, buf.stage_start, builder.every_search,
        builder.every_pe, builder.consensus, builder.device,
        builder.epoch_root,
        "ON (a build's epoch serves from the next warm draw)"
        if builder.adopt_lag else "OFF (a build's epoch serves its own propose)")
    return builder


# ---------------------------------------------------------------------------
# the move side: three containers, one per propose
# ---------------------------------------------------------------------------
def warm_seed(build_seed, tag) -> typing.Optional[int]:
    """``recipe._warm_start_seed`` on a bare build seed (same formula, pinned by test).

    Args:
        build_seed (int or None): The run's (per-rank) build seed.
        tag (int): The seed domain tag.

    Returns:
        int or None: A uint32 seed from ``SeedSequence([build_seed, tag])``;
        ``None`` when ``build_seed`` is ``None``.
    """
    if build_seed is None:
        return None
    return int(np.random.SeedSequence([int(build_seed), int(tag)])
               .generate_state(1, dtype=np.uint32)[0])


#: the three containers a warm move draws from, in split order
CHOICES = ("orig", "full", "filtered")


def parse_split(spec) -> typing.Tuple[float, float, float]:
    """``"orig,full,filtered"`` weights -> three floats (>= 0, not all 0).

    The parser of ``GB_WARM_SELF_SPLIT_SEARCH`` / ``GB_WARM_SELF_SPLIT_PE``;
    the weights need not sum to 1.

    Args:
        spec (str, tuple or list): Three comma-separated numbers, or a
            3-sequence.

    Returns:
        tuple[float, float, float]: The weights.

    Raises:
        ValueError: Not three finite non-negative numbers, or all zero.

    Example:
        >>> from lisatools.globalfit.warmstart.self_refit import parse_split
        >>> parse_split("0.5,0,0.5")
        (0.5, 0.0, 0.5)
    """
    parts = list(spec) if isinstance(spec, (tuple, list)) else str(spec).split(",")
    try:
        w = tuple(float(x) for x in parts)
    except (TypeError, ValueError):
        raise ValueError(f"warm-self split {spec!r}: three numbers "
                         "'original,full,filtered' expected") from None
    if (len(w) != 3 or not all(np.isfinite(x) and x >= 0.0 for x in w)
            or sum(w) <= 0.0):
        raise ValueError(f"warm-self split {spec!r}: three non-negative "
                         "weights 'original,full,filtered', not all zero")
    return w


class WarmSelfMixer:
    """The warm moves' RJ birth law: ORIGINAL, FULL self refit or FILTERED self refit, per propose.

    Duck-typed like :class:`.proposal.WarmStartComponents` (``rvs`` /
    ``logpdf``), delegating to the container chosen for the CURRENT
    propose. The choice changes only at :meth:`begin_propose` (head /
    single process) or :meth:`apply_directive` (a compute rank, from the
    head's directive), so every density of one propose -- the birth draws,
    the birth factors and the BandSorter's death factors -- comes from one
    container. Before any self epoch exists every propose uses the
    original; then each propose DRAWS its container at random with
    probabilities ~ ``split`` (original, full, filtered): ``1,1,1`` one third
    each, ``0.5,0,0.5`` the original on 50 % and the filtered one on 50 %,
    ``0,0,1`` always the filtered one. An epoch without a filtered npz
    serves its full refit where the filtered one is chosen. On the head,
    :meth:`begin_propose` first adopts any newer complete epoch on disk,
    then lets the registered head builder build a due epoch; under the
    default adoption lag (below) the propose that triggered the build
    still draws from the epoch it held, and the next warm draw adopts the
    new one. Pickle/deepcopy safe: numpy containers and plain scalars only.

    ``adopt_lag`` (default True; ``GB_WARM_SELF_ADOPT_LAG``, Mike
    2026-10-16) amends the build propose: an epoch the head builds INSIDE
    :meth:`begin_propose` is held (``held_epoch``) and that propose draws
    from the epoch adopted before the build; the next warm draw adopts it.
    ``adopt_lag=False``: the propose that triggered a build draws from it.

    Built by ``recipe._warm_self_wrap`` around each warm twin's container
    when ``GB_WARM_SELF=1`` (``rj_warm_search*`` takes the search split,
    every other name the PE split). In a stage that never refits
    (``gb_search_seed``, ``source_search``, ...) every propose uses the
    original, even when a self epoch exists.

    Args:
        original (WarmStartComponents): The parent warm-start container.
        epoch_root (str): :func:`self_epoch_root` of the store.
        load_kwargs (dict): :meth:`.proposal.WarmStartComponents.from_npz`
            options for the self epochs (the original's ``new_tobs``,
            ``use_cupy``, ``floor_box``, ``floor_eps``, ``p_floor`` and, for
            the PE twin, ``circ_images``).
        split (str or sequence, optional): "original,full,filtered"
            weights (:func:`parse_split`). Default ``(1, 1, 1)``.
        build_seed (int, optional): The run's per-rank build seed (epoch and
            choice seeds derive from it). Default ``None``.
        tag (int, optional): The twin's seed domain tag. Default ``0``.
        name (str, optional): The move name (log lines). Default ``"warm"``.
        adopt_lag (bool, optional): Default ``True``.

    Attributes:
        split (tuple[float, float, float]): The parsed weights.
        original, self_full, self_filtered: The three containers (the self
            ones ``None`` until an epoch is adopted; ``self_filtered`` also
            ``None`` for an epoch without a filtered npz).
        epoch (int or None): The adopted epoch.
        choice (str): This propose's container, one of :data:`CHOICES`.
        tallies (dict): Per-container propose / birth / death counts.
    """

    is_warm_self_mixer = True
    ndim = 9

    def __init__(self, original, *, epoch_root: str, load_kwargs: dict,
                 split=(1.0, 1.0, 1.0), build_seed=None, tag: int = 0,
                 name: str = "warm", adopt_lag: bool = True):
        self.split = parse_split(split)
        #: hold an epoch built at this propose's entry for the next draw
        self.adopt_lag = bool(adopt_lag)
        #: the epoch the head built in the CURRENT begin_propose and held back
        #: (adopted at the next draw: the adopt lag); None on every other
        #: propose / rank
        self.held_epoch: typing.Optional[int] = None
        self.original = original
        self.self_full = None
        self.self_filtered = None
        self.epoch: typing.Optional[int] = None
        self.epoch_info: dict = {}
        self.epoch_root = str(epoch_root)
        self.load_kwargs = dict(load_kwargs)
        self.build_seed = build_seed
        self.tag = int(tag)
        self.name = str(name)
        self.choice = "orig"
        #: proposes since a self container first existed
        self.n_self_window = 0
        #: the per-propose container draw (only the head draws; the ranks
        #: take its directive): probabilities ~ split, seeded per run + twin
        self._probs = np.asarray(self.split, dtype=float) / float(sum(self.split))
        self._choice_rng = np.random.default_rng(
            warm_seed(build_seed, int(tag) ^ _CHOICE_SEED_XOR))
        self._bad_epochs: set = set()
        self.tallies = {c: dict(proposes=0, births=0, viable=0, birth_acc=0,
                                viable_cold=0, birth_acc_cold=0, deaths=0,
                                death_acc=0) for c in CHOICES}

    # -- the active container ----------------------------------------------
    @property
    def self_container(self):
        """The FULL self refit (the pre-2026-10-15 name of the self container)."""
        return self.self_full

    @property
    def active(self):
        """This propose's container (filtered -> full -> original fallback)."""
        if self.choice == "filtered":
            if self.self_filtered is not None:
                return self.self_filtered
            if self.self_full is not None:
                return self.self_full
        elif self.choice == "full" and self.self_full is not None:
            return self.self_full
        return self.original

    @property
    def n_components(self) -> int:
        """The active container's component count."""
        return int(self.active.n_components)

    @property
    def use_cupy(self) -> bool:
        """Whether the active container returns device arrays."""
        return bool(getattr(self.active, "use_cupy", False))

    def rvs(self, size=1):
        """Birth draws from the active container (:meth:`.proposal.WarmStartComponents.rvs`)."""
        return self.active.rvs(size=size)

    def logpdf(self, x):
        """Log density of the active container (:meth:`.proposal.WarmStartComponents.logpdf`)."""
        return self.active.logpdf(x)

    # -- adoption -------------------------------------------------------------
    def epoch_seed(self, k: int) -> typing.Optional[int]:
        """``_warm_start_seed(gb_info, tag ^ (k + 1))``: per rank AND per epoch."""
        return warm_seed(self.build_seed, self.tag ^ (int(k) + 1))

    def filtered_seed(self, k: int) -> typing.Optional[int]:
        """The FILTERED container's seed: its own stream, per rank and epoch."""
        return warm_seed(self.build_seed,
                         self.tag ^ (int(k) + 1) ^ _FILTERED_SEED_XOR)

    def _adopt(self, k: int, manifest: typing.Optional[dict], rank) -> bool:
        """Load epoch ``k`` (both containers); False (and a warning) on failure."""
        from .proposal import WarmStartComponents

        if manifest is None:
            manifest = read_manifest(self.epoch_root, k)
        if manifest is None:
            if k not in self._bad_epochs:
                self._bad_epochs.add(k)
                logger.warning(
                    "[GB_WARM_SELF] rank %s (%s): epoch %d is not complete at "
                    "%s; keeping epoch %s.", rank, self.name, k,
                    epoch_dir(self.epoch_root, k), self.epoch)
            return False
        try:
            full = WarmStartComponents.from_npz(
                manifest["npz_path"], seed=self.epoch_seed(k),
                **self.load_kwargs)
            fpath = manifest.get("npz_filtered_path")
            filt = (None if not fpath else WarmStartComponents.from_npz(
                fpath, seed=self.filtered_seed(k), **self.load_kwargs))
        except Exception as exc:                  # noqa: BLE001
            if k not in self._bad_epochs:
                self._bad_epochs.add(k)
                logger.warning(
                    "[GB_WARM_SELF] rank %s (%s): epoch %d failed to load "
                    "(%s: %s); keeping epoch %s.", rank, self.name, k,
                    type(exc).__name__, exc, self.epoch)
            return False
        self.self_full = full
        self.self_filtered = filt
        self.epoch = int(k)
        self.epoch_info = {kk: manifest.get(kk) for kk in (
            "rows", "n_samples", "n_components", "built_at_row", "stage",
            "n_clusters", "n_clusters_filtered", "built_on")}
        rws = manifest.get("rows") or [None, None]
        logger.info(
            "[GB_WARM_SELF] rank %s adopted epoch %d (rows %s..%s, %s "
            "samples, %d components) into %s; filtered: %s; split "
            "original/full/filtered %s.", rank, k, rws[0], rws[-1],
            manifest.get("n_samples"), full.n_components, self.name,
            "none (serves the full refit)" if filt is None
            else f"{filt.n_components} components",
            "/".join(f"{w:g}" for w in self.split))
        return True

    def poll(self, rank=None) -> None:
        """Adopt the newest COMPLETE epoch if it is newer than the one held."""
        found = latest_complete_epoch(self.epoch_root, newer_than=self.epoch)
        if found is None:
            return
        k, man = found
        if k in self._bad_epochs:
            return
        self._adopt(k, man, rank)

    def _next_choice(self) -> str:
        """Draw this propose's container with probabilities ~ ``split``.

        Mike, 2026-10-15: "just draw each of these with 1/3 probability
        each". A fixed-probability mixture of exact MH kernels is itself a
        valid kernel; a zero weight is never drawn.
        """
        return CHOICES[int(self._choice_rng.choice(len(CHOICES), p=self._probs))]

    def begin_propose(self, rank=None, stage=None) -> dict:
        """HEAD / single process: poll, build if due, choose; the directive.

        The poll comes FIRST, so an epoch already complete on disk (the
        previous within-run refit, a resumed run's, a saver-built one) is
        adopted at once. An epoch the registered head builder builds in THIS
        call is, with ``adopt_lag`` (Mike, 2026-10-16: "use the previous
        information on the current proposal and then switch it out"), NOT
        adopted: this propose draws from the epoch held before the build
        (the original warm start when there is none) and the directive
        ships THAT epoch, so every rank uses it too; the next warm draw's
        poll adopts the new one. ``adopt_lag=False`` re-polls after the
        build and draws from the new epoch at once.

        Args:
            rank (int, optional): This rank (only ``None`` / 0 builds).
            stage (str, optional): The stage the propose runs in.

        Returns:
            dict: The directive ``{"epoch": int or None, "choice": str}``
            shipped to the compute ranks.
        """
        self.held_epoch = None
        self.poll(rank)
        builder = head_builder_for(self.epoch_root)
        if builder is not None and stage is not None and rank in (None, 0):
            if builder.maybe_build(stage):
                if self.adopt_lag:
                    self.held_epoch = getattr(builder, "last_built_epoch", None)
                else:
                    self.poll(rank)
        # A stage that never refits (gb_search_seed, source_search, ...)
        # draws only from the ORIGINAL parent warm start, even when a self
        # epoch exists on disk (Mike, 2026-10-15: "making sure gb search seed
        # is fixed to the old fit"). ``stage=None`` (no stage stamp) keeps
        # the epoch-based choice.
        if stage is not None and warm_self_stage_kind(stage) is None:
            self.choice = "orig"
        elif self.self_full is None:
            self.choice = "orig"
        else:
            self.n_self_window += 1
            self.choice = self._next_choice()
        return {"epoch": self.epoch, "choice": self.choice}

    def apply_directive(self, directive, rank=None) -> None:
        """COMPUTE RANK: adopt the head's epoch and take the head's choice.

        Idempotent on the head (its own directive). ``None`` -> original. A
        pre-2026-10-15 directive's ``"self"`` is the full refit. A rank that
        cannot load the head's epoch keeps what it holds and says so; its
        propose stays internally consistent either way.

        The directive is AUTHORITATIVE: a rank never polls the epoch root,
        so the shipped epoch is the only one it adopts, even when a newer
        complete one is on disk (the adoption lag ships the epoch held
        before a build while the fresh one already sits there), and a
        directive without an epoch (the head holds no self refit) is the
        original whatever the rank holds.

        Args:
            directive (dict or None): The head's :meth:`begin_propose` output.
            rank (int, optional): This rank (log lines).
        """
        if not directive:
            self.choice = "orig"
            return
        k = directive.get("epoch")
        if k is not None and k != self.epoch and int(k) not in self._bad_epochs:
            self._adopt(int(k), None, rank)
        c = str(directive.get("choice") or "orig")
        c = "full" if c == "self" else c
        if c not in CHOICES or k is None:
            c = "orig"
        self.choice = c if (c == "orig" or self.self_full is not None) \
            else "orig"

    # -- tallies ----------------------------------------------------------------
    def note_census(self, sp: typing.Optional[dict], *, move_name=None,
                    rank=None) -> None:
        """Fold this propose's ``[GB_ACCEPT rj-split]`` census into the chosen container's tally.

        Logs one ``[GB_WARM_SELF <move>]`` line with the cumulative tallies
        of all three containers.

        Args:
            sp (dict or None): The census (``births``, ``viable``,
                ``birth_acc``, ``viable_cold``, ``birth_acc_cold``,
                ``deaths``, ``death_acc``; missing keys count 0).
            move_name (str, optional): Log label. Default the mixer's name.
            rank (int, optional): Log label.
        """
        t = self.tallies[self.choice if self.choice in self.tallies else "orig"]
        t["proposes"] += 1
        for kk in ("births", "viable", "birth_acc", "viable_cold",
                   "birth_acc_cold", "deaths", "death_acc"):
            t[kk] += int((sp or {}).get(kk, 0) or 0)

        def _fmt(c):
            r = self.tallies[c]
            return (f"{c}: proposes {r['proposes']} births viable "
                    f"{r['viable']} acc {r['birth_acc']} "
                    f"({r['birth_acc'] / max(r['viable'], 1):.2e}) cold "
                    f"{r['birth_acc_cold']}/{r['viable_cold']} "
                    f"({r['birth_acc_cold'] / max(r['viable_cold'], 1):.2e})"
                    f" deaths acc {r['death_acc']}/{r['deaths']}")

        logger.info(
            "[GB_WARM_SELF %s] %sthis propose used %s%s | cumulative %s",
            move_name or self.name,
            "" if rank is None else f"r{rank} ",
            self.choice.upper(),
            f" (epoch {self.epoch})" if self.choice != "orig" else "",
            " || ".join(_fmt(c) for c in CHOICES))
