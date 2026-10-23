"""Automatic warm-start component building (user ruling 2026-09-14).

The warm-start refit proposal (``rj_warm_search`` / ``rj_warm_pe``) needs a
REFEREED components npz built from the previous run's store by the three-step
pipeline fit -> referee -> apply. The 6mo campaign's first launch failed on
that npz being missing, so :func:`ensure_warm_start_components` runs the
pipeline automatically at recipe build when the npz is absent and
``GB_WARM_START_SOURCE_STORE`` names the source store.

The pipeline stages are INSTALLED sibling modules of this one
(:mod:`.fit_from_store`, :mod:`.match_referee`, :mod:`.referee_apply` --
user ruling 2026-09-14: "All the warmstart code should be part of the
globalfit/lisatools package"), and each step is a plain in-process python
call to that module's ``main(argv)`` (earlier ruling the same day: "python
imports right? not like calling bash") -- no subprocess, no interpreter
spawn, no script files.

MPI-safe by a lock DIRECTORY next to the target: ``run_combined_staged.py``
builds on every rank before roles resolve, so one rank wins ``os.mkdir`` on
the lock and builds while the others poll for the npz. A crashed builder
releases the lock in ``finally``; a lock left behind by a killed process is
taken over once it is older than the timeout.

Where it sits: ``recipe.build_gb_moves`` calls
:func:`ensure_warm_start_components` on every rank with
``GB_WARM_START_COMPONENTS`` before either warm twin loads the npz. On the
9mo / 12mo launchers that path is ``${STORE_DIR}/warmstart/<parent>_refereed.npz``
inside the run directory, so it is built on the FIRST launch only and every
resume finds it (a ``.source`` sidecar records the store it was fitted
from, and a later launch against a different seed store logs a warning).
The same three-step runner (:func:`_default_runner`) also runs every
self-refit epoch (:func:`.self_refit.build_self_epoch`).

Env knobs (read here, not Settings fields -- they configure a one-shot
offline build, not the sampler):

* ``GB_WARM_START_SOURCE_STORE`` -- the previous run's FULL FINAL h5 (never a
  make_snapshots extract). Launchers: ``${GF_SEED_STORE}``.
* ``GB_WARM_START_LAST_K`` -- default 10; the number of the source store's
  last STORED ROWS fitted (x every walker). A parent ``full_pe`` stores one
  row per iteration; a numbered search stage stores 3 rows per cycle under
  ``GB_SEARCH_LEGS=1``. Launchers: 300.
* ``GB_WARM_START_SOURCE_TOBS`` -- the SOURCE store's Tobs [s], default
  7776000.0 = 3 months -- NOT this run's Tobs;
  ``WarmStartComponents.from_npz(new_tobs=...)`` does the rescale at load.
  Launchers: the parent's Tobs (15552000 on the 9mo, 23328000 on the 12mo).
* ``GB_WARM_START_BUILD_TIMEOUT`` -- seconds a waiting rank polls (and the
  age at which a lock is taken over as stale), default 7200. Launchers:
  unset.
"""

from __future__ import annotations

import logging
import os
import time
import typing

__all__ = ["ensure_warm_start_components"]

logger = logging.getLogger(__name__)

_STEPS = ("fit_from_store", "match_referee", "referee_apply")


def _manual_recipe(path: str, store: str, last_k: typing.Optional[int] = None,
                   tobs: typing.Optional[float] = None) -> str:
    """The three by-hand commands, with THIS build's last_k / source tobs.

    ``None`` = the environment, as the build itself resolves them (the hint
    used to print the 3mo-parent values 10 / 7776000 whatever the run was
    configured with; 2026-10-13). Used in the ``FileNotFoundError`` text of
    :func:`ensure_warm_start_components`.

    Args:
        path: The target refereed npz.
        store: The source store h5 (or a placeholder string).
        last_k: Stored rows to fit. Default ``None`` = ``GB_WARM_START_LAST_K``
            (else 10).
        tobs: Source Tobs [s]. Default ``None`` = ``GB_WARM_START_SOURCE_TOBS``
            (else 7776000).

    Returns:
        str: Three newline-separated ``python -m`` commands.
    """
    if last_k is None:
        last_k = int(os.environ.get("GB_WARM_START_LAST_K", "10"))
    if tobs is None:
        tobs = float(os.environ.get("GB_WARM_START_SOURCE_TOBS", "7776000.0"))
    stem = os.path.splitext(os.path.basename(path))[0]
    fit = os.path.join(os.path.dirname(path) or ".", stem + "_fit.npz")
    ref = os.path.splitext(fit)[0] + "_referee.npz"
    return (
        f"python -m lisatools.globalfit.warmstart.fit_from_store "
        f"--store {store} --last-k {int(last_k)} --tobs {float(tobs):.0f} --out {fit}\n"
        f"python -m lisatools.globalfit.warmstart.match_referee "
        f"--npz {fit} --store {store}\n"
        f"python -m lisatools.globalfit.warmstart.referee_apply "
        f"--fit {fit} --referee {ref} --out {path}"
    )


def _default_runner(cmd: typing.Sequence) -> None:
    """Run one pipeline step as a PLAIN PYTHON CALL to the sibling module's ``main(argv)``.

    The stage modules import lazily so the exists-check fast path of
    :func:`ensure_warm_start_components` stays light. Also the runner of
    every self-refit epoch (:func:`.self_refit.build_self_epoch`).

    Args:
        cmd: ``[step_name, *args]``; ``step_name`` is ``"fit_from_store"``,
            ``"match_referee"`` or ``"referee_apply"``; ``args`` are
            stringified and passed as ``argv``.

    Raises:
        RuntimeError: When the step raises ``SystemExit`` with a non-zero
            code (argparse errors, explicit exits).
        KeyError: For an unknown ``step_name``.
    """
    from . import fit_from_store, match_referee, referee_apply

    mods = {"fit_from_store": fit_from_store,
            "match_referee": match_referee,
            "referee_apply": referee_apply}
    step, argv = str(cmd[0]), [str(a) for a in cmd[1:]]
    logger.info("[WARMSTART-BUILD] running %s %s", step, " ".join(argv))
    try:
        mods[step].main(argv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(
                f"warmstart {step} exited with code {exc.code}."
            ) from exc


def _require(path: str, step: str) -> None:
    """Raise ``RuntimeError`` when ``step`` returned without writing ``path``."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"[WARMSTART-BUILD] {step} completed without producing "
            f"{path!r} -- read its output above."
        )


#: Sidecar recording which store a warm-start npz was fitted from.
_PROVENANCE_SUFFIX = ".source"


def _provenance_path(path: str) -> str:
    """``<path>.source``: the sidecar holding the absolute source-store path."""
    return str(path) + _PROVENANCE_SUFFIX


def _write_warm_start_provenance(path: str, store: str) -> None:
    """Record the source store next to a freshly built npz (best effort)."""
    try:
        with open(_provenance_path(path), "w") as fh:
            fh.write(os.path.abspath(store) + "\n")
    except OSError as exc:  # noqa: BLE001 -- provenance, never a failure
        logger.debug("warm start: could not write provenance (%r)", exc)


def _check_warm_start_provenance(path: str, store: str, log) -> None:
    """Warn when an EXISTING npz was fitted from a different store.

    ⚠ THE FAILURE THIS CATCHES (2026-09-24 audit). The npz path is a fixed
    filename inside the run store, and the auto-build fires only when that
    file is ABSENT. So relaunching the same ``STORE_DIR`` with a different
    seed store leaves the warm-start mixture fitted from the OLD run while
    the psd/galfor pin -- read live from the new one -- follows the NEW run.
    That is precisely the split the 2026-09-24 ruling forbids ("the warmstart
    and PSD/GB frozen start point come from the same folder"), and nothing
    else in the chain can see it: the two env vars still agree, the file
    still exists, and the fit still loads.

    A WARNING rather than a refusal: an npz written before this sidecar
    existed has no provenance to check, and a deliberate cross-seeding is a
    legitimate (if unusual) thing to ask for. The log line is what makes it
    a decision instead of an accident.

    Args:
        path: The existing refereed npz.
        store: This launch's source store (empty = no check).
        log (logging.Logger): Where the warning goes.
    """
    if not store:
        return
    rec = _provenance_path(path)
    try:
        with open(rec) as fh:
            was = fh.read().strip()
    except OSError:
        log.warning(
            "warm start: %s exists but carries no provenance record, so it "
            "CANNOT be verified against the current "
            "GB_WARM_START_SOURCE_STORE=%s. If this store was relaunched "
            "against a different seed run, the warm-start mixture is from "
            "the OLD one while the psd/galfor pin is from the new one -- "
            "delete the npz to force a rebuild.", path, store)
        return
    if was and os.path.abspath(was) != os.path.abspath(store):
        log.warning(
            "warm start: %s was fitted from\n    %s\nbut this launch's seed "
            "store is\n    %s\nThe warm-start mixture and the psd/galfor "
            "pin would then come from DIFFERENT runs. Delete the npz (and "
            "its %s sidecar) to refit from the current seed, or point "
            "GF_SEED_STORE back at the original.",
            path, was, store, _PROVENANCE_SUFFIX)


def ensure_warm_start_components(
    path: str,
    *,
    store: typing.Optional[str] = None,
    last_k: typing.Optional[int] = None,
    tobs: typing.Optional[float] = None,
    timeout: typing.Optional[float] = None,
    poll: float = 5.0,
    runner: typing.Optional[typing.Callable[[list], None]] = None,
    log: typing.Optional[logging.Logger] = None,
) -> str:
    """The refereed warm-start npz at ``path`` -- built if missing.

    When the file already exists this is a pure check (plus the provenance
    warning of :func:`_check_warm_start_provenance`). Otherwise the fit ->
    referee -> apply pipeline runs against ``store`` (or
    ``GB_WARM_START_SOURCE_STORE``), under a lock directory
    ``<path>.build.lock`` so that concurrent MPI ranks build exactly once;
    the output lands atomically (``<path>.building.npz`` + ``os.replace``),
    so a waiting rank never sees a partial npz. The intermediates
    ``<stem>_fit.npz`` and ``<stem>_fit_referee.npz`` stay next to ``path``,
    and ``<path>.source`` records the source store. The fit step runs with
    the fitter's CLI defaults otherwise (observable basis, ``--gmm-gpu
    auto``), the referee with its defaults (``--backend`` from
    ``GB_WARM_START_REFEREE_BACKEND``, default ``cpu``).

    Args:
        path: The refereed npz (``GB_WARM_START_COMPONENTS``).
        store: Source store h5. Default ``None`` = ``GB_WARM_START_SOURCE_STORE``.
        last_k: Source rows to fit (STORED ROWS x every walker). Default
            ``None`` = ``GB_WARM_START_LAST_K`` (else 10).
        tobs: The SOURCE store's Tobs [s]. Default ``None`` =
            ``GB_WARM_START_SOURCE_TOBS`` (else 7776000.0).
        timeout: Seconds a waiting rank polls before giving up, and the age
            [s] of a lock taken over as stale. Default ``None`` =
            ``GB_WARM_START_BUILD_TIMEOUT`` (else 7200).
        poll: Seconds between a waiting rank's checks. Default ``5.0``.
        runner: ``runner([step, *argv])`` runs one step. Default ``None`` =
            :func:`_default_runner` (in-process ``main(argv)``).
        log: Logger. Default ``None`` = this module's.

    Returns:
        str: ``path``.

    Raises:
        FileNotFoundError: When the npz is missing and no source store is
            set (the message carries the three by-hand commands), or when
            the source store does not exist.
        RuntimeError: When a waiting rank times out, or a step fails or
            writes no output.

    Example:
        .. code-block:: python

            from lisatools.globalfit.warmstart import ensure_warm_start_components

            npz = ensure_warm_start_components(
                "gf_prod_9mo/warmstart/gf_prod_6mo_v9_4gpu_refereed.npz",
                store="gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5",
                last_k=300, tobs=15552000.0)
    """
    log = log or logger
    store = (store or os.environ.get("GB_WARM_START_SOURCE_STORE", "")
             or "").strip()
    if os.path.exists(path):
        _check_warm_start_provenance(path, store, log)
        return path

    if not store:
        raise FileNotFoundError(
            f"warm-start components {path!r} do not exist and "
            "GB_WARM_START_SOURCE_STORE is not set, so they cannot be "
            "built automatically. Either set GB_WARM_START_SOURCE_STORE "
            "to the previous run's FULL FINAL store h5 (never a "
            "make_snapshots extract) or build the npz by hand:\n"
            + _manual_recipe(path, "<previous-run store h5>", last_k, tobs)
        )
    if not os.path.exists(store):
        raise FileNotFoundError(
            f"warm-start source store {store!r} does not exist "
            "(GB_WARM_START_SOURCE_STORE)."
        )

    last_k = int(last_k if last_k is not None
                 else os.environ.get("GB_WARM_START_LAST_K", "10"))
    tobs = float(tobs if tobs is not None
                 else os.environ.get("GB_WARM_START_SOURCE_TOBS",
                                     "7776000.0"))
    timeout = float(timeout if timeout is not None
                    else os.environ.get("GB_WARM_START_BUILD_TIMEOUT",
                                        "7200"))
    runner = runner or _default_runner

    workdir = os.path.dirname(path) or "."
    os.makedirs(workdir, exist_ok=True)
    lock = path + ".build.lock"

    try:
        os.mkdir(lock)
    except FileExistsError:
        # another rank (or process) is building; wait for the npz.
        log.info(
            "[WARMSTART-BUILD] %s is being built elsewhere (lock %s); "
            "waiting up to %.0f s.", path, lock, timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(path):
                return path
            # stale lock (builder killed without cleanup): take over.
            try:
                if time.time() - os.path.getmtime(lock) > timeout:
                    os.rmdir(lock)
                    return ensure_warm_start_components(
                        path, store=store, last_k=last_k, tobs=tobs,
                        timeout=timeout, poll=poll, runner=runner, log=log)
            except OSError:
                pass
            time.sleep(poll)
        raise RuntimeError(
            f"timed out after {timeout:.0f} s waiting for another process "
            f"to build {path!r} (lock {lock}). If no builder is alive, "
            "remove the lock directory and relaunch."
        )

    stem = os.path.splitext(os.path.basename(path))[0]
    fit_npz = os.path.join(workdir, stem + "_fit.npz")
    referee_npz = os.path.splitext(fit_npz)[0] + "_referee.npz"
    tmp_out = path + ".building.npz"
    try:
        log.warning(
            "[WARMSTART-BUILD] %s missing -- building it now from %s "
            "(last_k=%d, source tobs=%.0f s). fit -> referee -> apply; "
            "other ranks wait on the lock.", path, store, last_k, tobs)
        t0 = time.perf_counter()
        runner([_STEPS[0], "--store", store, "--last-k", str(last_k),
                "--tobs", str(tobs), "--out", fit_npz])
        _require(fit_npz, "fit_from_store")
        runner([_STEPS[1], "--npz", fit_npz, "--store", store])
        _require(referee_npz, "match_referee")
        runner([_STEPS[2], "--fit", fit_npz, "--referee", referee_npz,
                "--out", tmp_out])
        _require(tmp_out, "referee_apply")
        os.replace(tmp_out, path)
        # Record WHICH store this was fitted from, so a later launch of the
        # same run directory against a different seed can be caught (see
        # _check_warm_start_provenance).
        _write_warm_start_provenance(path, store)
        log.warning(
            "[WARMSTART-BUILD] built %s in %.1f s (source %s).",
            path, time.perf_counter() - t0, store)
        return path
    finally:
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        try:
            os.rmdir(lock)
        except OSError:
            pass
