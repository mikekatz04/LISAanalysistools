"""Mid-iteration checkpointing for the global fit (preemption protection).

Motivation (2026-08-27, the 6-mo sources probe): on the spot partition a
job can be preempted every 30--75 minutes while a single global-fit
iteration (MBH ~26 min + EMRI ~37 min + ...) takes longer than that
window, so the run never reaches its first ``save_step`` and every
relaunch is a fresh start. This module persists the in-memory
:class:`~lisatools.globalfit.state.GFState` at safe sub-iteration
boundaries -- between the wrapped moves of a
:class:`~lisatools.globalfit.moves.globalfitmove.GFCombineMove` stage and
between leaves of the per-leaf add/remove PE moves -- so a relaunch
resumes from the newest snapshot instead of losing the whole iteration.

Semantics
---------
A checkpoint is a full, coherent state snapshot. Resuming from it simply
STARTS A NEW ITERATION from that state: MCMC needs no replay of the
interrupted iteration's remaining moves, because every accepted sub-move
is already folded into the snapshot (and re-proposing a branch that
already ran is just more valid sampling). The ``stored_iteration``
counter in the meta block decides precedence at resume time: the
checkpoint wins over the HDF store iff it was written at (or after) the
store's last stored iteration; an older checkpoint is stale and ignored
(the store already contains everything it knew).

The module is a per-process singleton armed by ``run.py`` on the
sampling rank only (``GeneralSettings.midit_checkpoint``; env
``MIDIT_CHECKPOINT``). :func:`maybe_write` is throttled
(``midit_checkpoint_min_interval`` seconds between writes; env
``MIDIT_CHECKPOINT_MIN_INTERVAL``) so hooks can sit at every boundary at
negligible cost, and it never raises -- a checkpoint failure must not
kill the run. Writes are atomic (tmp + fsync + ``os.replace``), so a
kill during a write leaves the previous checkpoint intact.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
import typing

__all__ = [
    "arm",
    "armed",
    "checkpoint_path",
    "disarm",
    "load_for_resume",
    "maybe_write",
    "note_saved",
    "self_test",
]

logger = logging.getLogger(__name__)

#: bump on any change to the payload layout; a mismatched file is rejected.
MIDIT_CKPT_FORMAT = 1

_STATE: typing.Dict[str, typing.Any] = {
    "enabled": False,
    "path": None,
    "main_path": None,
    "min_interval": 600.0,
    "last_write": 0.0,
    "stored_iteration": 0,
}


def checkpoint_path(main_file_path: str) -> str:
    """Sidecar checkpoint path for a main backend file."""
    base, _ = os.path.splitext(main_file_path)
    return base + "_midit_checkpoint.pkl"


def arm(
    main_file_path: str,
    min_interval: float = 600.0,
    stored_iteration: int = 0,
) -> None:
    """Enable checkpointing for this process.

    Args:
        main_file_path: The run's main HDF backend path (the checkpoint
            lives next to it).
        min_interval: Minimum seconds between checkpoint writes. Hook
            calls inside the throttle window return immediately, so the
            worst-case overhead is one state pickle per interval. The
            clock starts at arm time (the first write is no earlier than
            ``min_interval`` after arming).
        stored_iteration: The backend's stored iteration count at arm
            time; :func:`note_saved` ticks it on every subsequent
            ``save_step`` so checkpoints record how much of the chain the
            store already holds.
    """
    _STATE["enabled"] = True
    _STATE["path"] = checkpoint_path(main_file_path)
    _STATE["main_path"] = main_file_path
    _STATE["min_interval"] = float(min_interval)
    _STATE["last_write"] = time.monotonic()
    _STATE["stored_iteration"] = int(stored_iteration)
    logger.info(
        "[MIDIT_CKPT] armed: %s (min interval %.0f s, store at iteration %d)",
        _STATE["path"], _STATE["min_interval"], _STATE["stored_iteration"],
    )


def disarm() -> None:
    """Disable checkpointing (e.g. for tests)."""
    _STATE["enabled"] = False
    _STATE["path"] = None


def armed() -> bool:
    """Whether this process writes mid-iteration checkpoints."""
    return bool(_STATE["enabled"])


def note_saved() -> None:
    """Record that a real ``save_step`` landed (ticks the stored count).

    Called by :meth:`GFHDFBackend.save_step` on the sampling rank in both
    save modes (sync write and saver-rank handoff). After this tick, a
    LATER checkpoint carries the new count, while the checkpoint written
    during the just-stored iteration becomes stale relative to a store
    that outlives it -- exactly the precedence :func:`load_for_resume`
    applies.
    """
    if _STATE["enabled"]:
        _STATE["stored_iteration"] += 1


def maybe_write(
    state,
    tag: str = "",
    prepare: typing.Optional[typing.Callable] = None,
    force: bool = False,
) -> bool:
    """Write a checkpoint if armed and the throttle interval has elapsed.

    Args:
        state: The state snapshot to persist. It must be coherent as a
            resume point: main-state cold rows in agreement with the
            sub-states (use ``prepare`` when the caller has to sync
            first) and every accepted change folded in. ``log_like`` /
            ``log_prior`` staleness is fine -- resume recomputes both
            from the rebuilt residual.
        tag: Short human-readable boundary label for the log/meta (e.g.
            ``"mbh leaf 3"``).
        prepare: Optional callable run on ``state`` ONLY when a write is
            actually due (outside the throttle window) -- the hook for
            work that is too expensive or mutating to do on every
            boundary, e.g. ``ResidualAddOneRemoveOneMove._sync_cold_row``.
        force: Skip the throttle (tests / final-boundary writes).

    Returns:
        True iff a checkpoint file was written.
    """
    if not _STATE["enabled"]:
        return False
    now = time.monotonic()
    if not force and now - _STATE["last_write"] < _STATE["min_interval"]:
        return False
    tmp = _STATE["path"] + ".tmp"
    try:
        if prepare is not None:
            prepare(state)
        meta = {
            "format": MIDIT_CKPT_FORMAT,
            "stored_iteration": int(_STATE["stored_iteration"]),
            "tag": str(tag),
            "unix_time": time.time(),
        }
        t0 = time.perf_counter()
        with open(tmp, "wb") as fh:
            pickle.dump(
                {"meta": meta, "state": state},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, _STATE["path"])
        _STATE["last_write"] = time.monotonic()
        logger.info(
            "[MIDIT_CKPT] wrote %s (%.1f MB, %.2f s) at stored iteration %d "
            "(boundary: %s)",
            _STATE["path"],
            os.path.getsize(_STATE["path"]) / 1e6,
            time.perf_counter() - t0,
            meta["stored_iteration"],
            tag or "?",
        )
        return True
    except Exception:  # noqa: BLE001 -- a checkpoint must never kill the run
        logger.exception(
            "[MIDIT_CKPT] write FAILED at boundary %r (run continues; the "
            "previous checkpoint, if any, is untouched)", tag,
        )
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def self_test(state, validate=None, logger_=None) -> bool:
    """Prove the whole checkpoint path works, at run start, on real state.

    Writes a forced checkpoint of ``state`` and reads it straight back
    through the REAL resume entry point (:func:`load_for_resume`, including
    the config-compatibility ``validate`` gate), so a run learns within its
    first minute whether preemption protection is actually functional --
    instead of discovering a broken pickle at the first boundary half an
    hour in, or (worse) discovering at relaunch that nothing was ever
    written.

    This is the standing answer to "has the mechanism been tested?" for
    every environment the tests cannot reach (GPU nodes, MPI layouts, new
    branch combinations): the run tests itself, in situ, at full scale.

    Failure is reported LOUDLY but is never fatal: checkpointing is
    protection, not science, and a run that cannot checkpoint is exactly
    the run we had before the feature existed.

    Args:
        state: The fully-built state the run is about to sample from.
        validate: The same ``state -> (ok, reason)`` gate a resume would
            apply. Self-rejection here is a BUG (the gate refusing the
            very state the run configured) and is logged as such.
        logger_: Logger for the verdict lines.

    Returns:
        True iff the write and the read-back both succeeded.
    """
    log = logger_ or logger
    if not _STATE["enabled"]:
        return False
    if not maybe_write(state, tag="startup self-test", force=True):
        log.error(
            "[MIDIT_CKPT] SELF-TEST FAILED at the WRITE step: this run has "
            "NO preemption protection (it will still sample normally). See "
            "the traceback above -- most likely something in the state is "
            "not picklable."
        )
        return False
    got = load_for_resume(
        _STATE["main_path"],
        stored_iteration=int(_STATE["stored_iteration"]),
        validate=validate,
        logger_=log,
    )
    if got is None:
        log.error(
            "[MIDIT_CKPT] SELF-TEST FAILED at the READ-BACK step: the "
            "checkpoint just written could not be reloaded (see the reason "
            "logged above). If that reason is a config mismatch, the "
            "validate gate is rejecting the run's OWN state -- a bug in the "
            "gate, not in the run. This run has NO preemption protection."
        )
        return False
    log.info(
        "[MIDIT_CKPT] SELF-TEST PASSED: wrote and reloaded %s. Preemption "
        "protection is ACTIVE -- checkpoints land at each PE leaf / stage "
        "sub-move boundary, at most one per %.0f s.",
        _STATE["path"], _STATE["min_interval"],
    )
    return True


def _reject(path: str) -> None:
    """Move a bad checkpoint aside (kept for post-mortem, never reloaded)."""
    try:
        os.replace(path, path + ".rejected")
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass


def load_for_resume(
    main_file_path: str,
    stored_iteration: int,
    validate: typing.Optional[typing.Callable] = None,
    logger_: typing.Optional[logging.Logger] = None,
):
    """Load the checkpoint next to ``main_file_path`` if it should win.

    Args:
        main_file_path: The run's main HDF backend path.
        stored_iteration: The store's current stored iteration count (0
            for a missing/empty store). A checkpoint written BEFORE the
            store's newest sample is stale and ignored.
        validate: Optional ``state -> (ok, reason)`` config-compatibility
            gate (branch set, shapes, ladders, band grids). A rejected
            checkpoint is moved aside to ``*.rejected`` -- resuming a
            reconfigured run from an incompatible snapshot must fail
            SAFE (fresh start), never crash mid-run at the first save.
        logger_: Logger for the resume messages (defaults to the module
            logger).

    Returns:
        ``(state, meta)`` when the checkpoint is usable, else ``None``.
    """
    log = logger_ or logger
    path = checkpoint_path(main_file_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        meta = payload["meta"]
        state = payload["state"]
        if int(meta.get("format", -1)) != MIDIT_CKPT_FORMAT:
            raise ValueError(
                f"checkpoint format {meta.get('format')!r} != "
                f"{MIDIT_CKPT_FORMAT}"
            )
    except Exception as exc:  # noqa: BLE001 -- unreadable file, fail safe
        log.warning(
            "[MIDIT_CKPT] %s is unreadable (%s: %s); moved aside to "
            "*.rejected and resuming without it.",
            path, type(exc).__name__, exc,
        )
        _reject(path)
        return None
    ck_it = int(meta.get("stored_iteration", -1))
    if ck_it < int(stored_iteration):
        log.info(
            "[MIDIT_CKPT] checkpoint at stored iteration %d is OLDER than "
            "the store (%d): stale, ignoring it (the store already contains "
            "everything it knew).", ck_it, int(stored_iteration),
        )
        return None
    if validate is not None:
        ok, why = validate(state)
        if not ok:
            log.warning(
                "[MIDIT_CKPT] checkpoint %s REJECTED (config mismatch: %s); "
                "moved aside to *.rejected and resuming without it.",
                path, why,
            )
            _reject(path)
            return None
    return state, meta
