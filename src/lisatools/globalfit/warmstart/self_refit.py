"""GB warm-start SELF refit: the run refits its warm-start mixture from its own store.

USER RULINGS (Mike, 2026-10-13):

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

WHY. Per-walker GB completeness at SNR > 8 was 91 % on the 9mo run while
the UNION of the four cold walkers was 96.5 %: each walker lacked ~180-220
SNR 6-12 leaves another walker held, and the PE cold RJ acceptance
(1e-4 .. 1e-3) never closed that. A mixture fitted to the last rows of ALL
cold walkers proposes to each walker what the others hold: a cross-walker
GB exchange through the existing warm-start machinery.

THE PIECES (all in this module, wired from three places):

1. **The build** (:class:`WarmSelfHook`, saver rank, ``hdfbackend``'s save
   loop). Right after one of its own saves, when the run's active recipe
   stage is ``gb_search_N`` (N >= 2) or ``full_pe`` and the stage's cadence
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
3. **The alternation** (:class:`WarmSelfMixer`, the warm moves'
   ``rj_proposal_distribution``). It holds the ORIGINAL parent-derived
   container and the SELF one (``None`` until an epoch exists). Each
   propose picks one container by a deterministic Bresenham cadence at
   ``GB_WARM_SELF_FRACTION`` (0.75 -> exactly 3 proposes in 4 use the self
   mixture; 100 % original while no self epoch exists) and that container
   serves the birth draws AND every logpdf of the propose (the BandSorter's
   death factors are computed from the same object), so forward and
   reverse densities always agree within a propose. A deterministic
   schedule of exact MH kernels is a valid (time-inhomogeneous) mixture.
4. **Adoption** (the head polls, every rank loads): at each propose the
   head's mixer stats the epoch root for a newer COMPLETE epoch and ships
   ``{"epoch", "choice"}`` to the compute ranks with the propose (the
   F-stat epoch directive pattern); each rank loads the epoch's npz from
   the shared filesystem into its own self container with the original's
   Tobs / floor box / floor_eps / p_floor / circ_images / use_cupy and the
   per-rank seed ``_warm_start_seed(gb_info, tag ^ (epoch + 1))``, so the
   candidate streams change per epoch and differ per rank.
5. **Resume**: nothing is held in memory that the disk does not also
   hold. The builder recomputes its clock from the store's recipe group
   and the latest complete epoch's ``built_at_row`` at every save; the
   moves adopt the latest complete epoch at their first propose.

Log tag: ``[GB_WARM_SELF]`` (build, adoption, the per-propose container
choice and the per-container birth tallies, the watchdog).
"""

from __future__ import annotations

import fractions
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
    "MANIFEST",
    "REFEREED_NPZ",
    "WarmSelfHook",
    "WarmSelfMixer",
    "WarmSelfWatchdog",
    "build_self_epoch",
    "copy_leaf_rows",
    "latest_complete_epoch",
    "make_warm_self_hook",
    "next_epoch_index",
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
_EPOCH_RE = re.compile(r"^epoch_(\d{4,})$")


# ---------------------------------------------------------------------------
# the epoch directory protocol
# ---------------------------------------------------------------------------
def self_epoch_root(store_path: str) -> str:
    """``<run_dir>/warmstart/self``: ONE definition for the saver and the moves.

    The run dir is the store's directory (``general_info.main_file_path`` on
    the moves, ``gb_reader.filename`` on the saver -- the same path).
    """
    return os.path.join(os.path.dirname(os.path.abspath(str(store_path))),
                        "warmstart", "self")


def epoch_dir(root: str, k: int) -> str:
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
    crashed; an unreadable one; a missing npz) is not an epoch.
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
    """``"search"`` for ``gb_search_N`` with N >= 2, ``"pe"`` for ``full_pe``, else None.

    "Starting in gb search 2 (and in full pe)": the numbered search stages
    from 2 on take the search cadence, ``full_pe`` the PE cadence. Every
    other stage (``gb_search_seed``, ``source_search``, ``noise_search``,
    ``gb_search_1``, ``replica_pe``) never builds.
    """
    if not stage:
        return None
    s = str(stage)
    if s == "full_pe":
        return "pe"
    m = _SEARCH_STAGE_RE.match(s)
    if m and int(m.group(1)) >= 2:
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
    Returns the provenance (source rows ``a..b`` inclusive, counts).
    """
    import h5py

    chain = g["chain"][branch]
    inds = g["inds"][branch]
    ll = g["log_like"]
    rows = int(rows)
    if rows <= 0:
        raise ValueError("the store holds no rows yet")
    a = max(0, rows - int(last_k))
    idx = list(range(a, rows))
    tail = tuple(chain.shape[1:-3])            # (nsamplers, ntemps) or (ntemps,)
    cold = (0,) * len(tail)
    nw, ndim = int(chain.shape[-3]), int(chain.shape[-1])
    packed, n_alive = [], 0
    for it in idx:
        c_it = np.asarray(chain[(it,) + cold])         # (nw, nleaves, ndim)
        i_it = np.asarray(inds[(it,) + cold], dtype=bool)
        per_w = []
        for w in range(nw):
            sel = np.flatnonzero(i_it[w])
            per_w.append(np.asarray(c_it[w, sel], dtype=np.float64))
            n_alive = max(n_alive, len(sel))
        packed.append(per_w)
    M = max(int(n_alive), 1)
    K = len(idx)
    ch = np.zeros((K,) + (1,) * len(tail) + (nw, M, ndim), dtype=np.float64)
    ind = np.zeros((K,) + (1,) * len(tail) + (nw, M), dtype=bool)
    n_leaf_rows = 0
    for r, per_w in enumerate(packed):
        for w, x in enumerate(per_w):
            n = len(x)
            ch[(r,) + cold + (w, slice(0, n))] = x
            ind[(r,) + cold + (w, slice(0, n))] = True
            n_leaf_rows += n
    llv = np.asarray(ll[a:rows])[(slice(None),) + cold + (slice(None),)]
    llv = llv.reshape((K,) + (1,) * len(tail) + (nw,))
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        gg = f.create_group("global_fit")
        gg.attrs["iteration"] = K
        gg.attrs["source_rows"] = [int(a), int(rows - 1)]
        gg.create_group("chain").create_dataset(branch, data=ch,
                                                compression="gzip")
        gg.create_group("inds").create_dataset(branch, data=ind,
                                               compression="gzip")
        gg.create_dataset("log_like", data=llv)
    os.replace(tmp, out_path)
    return dict(rows=[int(a), int(rows - 1)], n_rows=K, n_walkers=nw,
                n_samples=K * nw, n_leaf_rows=int(n_leaf_rows),
                max_alive=int(M))


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
    ``--gmm-gpu``: the saver's device, ``cpu`` without one -- the parent
    build's ``auto`` would take device 0 of whatever the process sees),
    ``match_referee --npz F --store S`` and ``referee_apply --fit F
    --referee R --out P``; every other option at its CLI default, i.e. the
    production warm start's settings. ``tobs`` is the RUN's own Tobs.
    Returns the walls and counts; raises on any failure (the hook catches).
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


class WarmSelfWatchdog:
    """The monitor hook's watchdog pattern, for the self-refit build.

    Same three signals as :class:`..monitor.hooks.MonitorWatchdog` (a save
    queued when the build finished; the build ate more than
    ``GF_MONITOR_WARN_FRAC`` of the measured save interval; states dropped
    across the build), reported under ``[GB_WARM_SELF]``. Dropped states
    switch the BUILDER off for the rest of the run (the moves keep the
    epochs they hold): the refit is never worth stored iterations.
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
        return self._inner.save_interval

    def note_save(self) -> None:
        self._inner.note_save()

    def report(self, elapsed: float, *, queued_after: bool,
               dropped_delta: int) -> None:
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
    saver's own backend. Never raises into the saver loop.
    """

    def __init__(self, *, epoch_root: str, tobs: float, last_k: int,
                 every_search: int, every_pe: int,
                 device: typing.Optional[int] = None, branch: str = "gb",
                 builder=None, runner=None):
        self.epoch_root = str(epoch_root)
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
        self.watchdog.note_save()

    def __call__(self, gb_reader, comm, main_rank, i, *,
                 dropped_so_far: int = 0) -> None:
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


def make_warm_self_hook(curr, *, store_path: typing.Optional[str] = None):
    """The saver's :class:`WarmSelfHook`, or ``None`` when the refit is off.

    Off unless ``GBSettings.warm_self`` (``GB_WARM_SELF=1``) AND the run has
    a parent warm start (``warm_start_components`` /
    ``GB_WARM_START_COMPONENTS``): the self epochs only feed the warm moves,
    which exist only with one.
    """
    gb = (getattr(curr, "source_info", None) or {}).get("gb")
    if gb is None or not bool(getattr(gb, "warm_self", False)):
        return None
    warm = str(getattr(gb, "warm_start_components", "")
               or os.environ.get("GB_WARM_START_COMPONENTS", "") or "").strip()
    if not warm:
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
# the move side: two containers, one per propose
# ---------------------------------------------------------------------------
def warm_seed(build_seed, tag) -> typing.Optional[int]:
    """``recipe._warm_start_seed`` on a bare build seed (same formula, pinned by test)."""
    if build_seed is None:
        return None
    return int(np.random.SeedSequence([int(build_seed), int(tag)])
               .generate_state(1, dtype=np.uint32)[0])


def _bresenham_self(n: int, frac: fractions.Fraction) -> bool:
    """Propose ``n`` (1-based) uses the self container: ``floor(n f) > floor((n-1) f)``.

    Exactly ``round-down(N f)`` of any N consecutive proposes from the
    first; 3/4 -> orig, self, self, self, orig, ...
    """
    p, q = frac.numerator, frac.denominator
    return (n * p) // q > ((n - 1) * p) // q


class WarmSelfMixer:
    """The warm moves' RJ birth law: the ORIGINAL warm start or the SELF refit, per propose.

    Duck-typed like :class:`.proposal.WarmStartComponents` (``rvs`` /
    ``logpdf``), delegating to the container chosen for the CURRENT
    propose. The choice changes only at :meth:`begin_propose` (head /
    single process) or :meth:`apply_directive` (a compute rank, from the
    head's directive), so every density of one propose -- the birth draws,
    the birth factors and the BandSorter's death factors -- comes from one
    container. Pickle/deepcopy safe: numpy containers, plain scalars, a
    ``Fraction``.
    """

    is_warm_self_mixer = True
    ndim = 9

    def __init__(self, original, *, epoch_root: str, fraction: float,
                 load_kwargs: dict, build_seed=None, tag: int = 0,
                 name: str = "warm"):
        f = float(fraction)
        if not (0.0 <= f <= 1.0):
            raise ValueError(
                f"GB_WARM_SELF_FRACTION must lie in [0, 1], got {fraction!r}")
        self.original = original
        self.self_container = None
        self.epoch: typing.Optional[int] = None
        self.epoch_info: dict = {}
        self.epoch_root = str(epoch_root)
        self.fraction = fractions.Fraction(f).limit_denominator(1000)
        self.load_kwargs = dict(load_kwargs)
        self.build_seed = build_seed
        self.tag = int(tag)
        self.name = str(name)
        self.choice = "orig"
        #: proposes since a self container first existed (the cadence index)
        self.n_self_window = 0
        self._bad_epochs: set = set()
        self.tallies = {c: dict(proposes=0, births=0, viable=0, birth_acc=0,
                                viable_cold=0, birth_acc_cold=0, deaths=0,
                                death_acc=0) for c in ("orig", "self")}

    # -- the active container ----------------------------------------------
    @property
    def active(self):
        if self.choice == "self" and self.self_container is not None:
            return self.self_container
        return self.original

    @property
    def n_components(self) -> int:
        return int(self.active.n_components)

    @property
    def use_cupy(self) -> bool:
        return bool(getattr(self.active, "use_cupy", False))

    def rvs(self, size=1):
        return self.active.rvs(size=size)

    def logpdf(self, x):
        return self.active.logpdf(x)

    # -- adoption -------------------------------------------------------------
    def epoch_seed(self, k: int) -> typing.Optional[int]:
        """``_warm_start_seed(gb_info, tag ^ (k + 1))``: per rank AND per epoch."""
        return warm_seed(self.build_seed, self.tag ^ (int(k) + 1))

    def _adopt(self, k: int, manifest: typing.Optional[dict], rank) -> bool:
        """Load epoch ``k`` into the self container; False (and a warning) on failure."""
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
            cont = WarmStartComponents.from_npz(
                manifest["npz_path"], seed=self.epoch_seed(k),
                **self.load_kwargs)
        except Exception as exc:                  # noqa: BLE001
            if k not in self._bad_epochs:
                self._bad_epochs.add(k)
                logger.warning(
                    "[GB_WARM_SELF] rank %s (%s): epoch %d failed to load "
                    "(%s: %s); keeping epoch %s.", rank, self.name, k,
                    type(exc).__name__, exc, self.epoch)
            return False
        self.self_container = cont
        self.epoch = int(k)
        self.epoch_info = {kk: manifest.get(kk) for kk in (
            "rows", "n_samples", "n_components", "built_at_row", "stage")}
        rws = manifest.get("rows") or [None, None]
        logger.info(
            "[GB_WARM_SELF] rank %s adopted epoch %d (rows %s..%s, %s "
            "samples, %d components) into %s; proposes use it %s of the "
            "time, the original warm start the rest.", rank, k, rws[0],
            rws[-1], manifest.get("n_samples"), cont.n_components, self.name,
            self.fraction)
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

    def begin_propose(self, rank=None) -> dict:
        """HEAD / single process: poll, choose this propose's container, return the directive."""
        self.poll(rank)
        if self.self_container is None:
            self.choice = "orig"
        else:
            self.n_self_window += 1
            self.choice = ("self" if _bresenham_self(self.n_self_window,
                                                     self.fraction)
                           else "orig")
        return {"epoch": self.epoch, "choice": self.choice}

    def apply_directive(self, directive, rank=None) -> None:
        """COMPUTE RANK: adopt the head's epoch and take the head's choice.

        Idempotent on the head (its own directive). ``None`` -> original.
        A rank that cannot load the head's epoch keeps what it holds and
        says so; its propose stays internally consistent either way.
        """
        if not directive:
            self.choice = "orig"
            return
        k = directive.get("epoch")
        if k is not None and k != self.epoch and int(k) not in self._bad_epochs:
            self._adopt(int(k), None, rank)
        c = str(directive.get("choice") or "orig")
        self.choice = c if (c == "orig" or self.self_container is not None) \
            else "orig"

    # -- tallies ----------------------------------------------------------------
    def note_census(self, sp: typing.Optional[dict], *, move_name=None,
                    rank=None) -> None:
        """Fold this propose's ``[GB_ACCEPT rj-split]`` census into the chosen container's tally."""
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
            "[GB_WARM_SELF %s] %sthis propose used %s%s | cumulative %s || %s",
            move_name or self.name,
            "" if rank is None else f"r{rank} ",
            self.choice.upper(),
            f" (epoch {self.epoch})" if self.choice == "self" else "",
            _fmt("orig"), _fmt("self"))
