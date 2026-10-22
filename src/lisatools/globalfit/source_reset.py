"""Push a RESUMED run's fixed-leaf source branches back to the catalogue injection.

Mike 2026-10-22: "We want those [the non-GB sources: VGB, MBHB, SOBBH, EMRI]
to always start at the catalog injected values rather than the last run's
values ... make a script that pushes the ongoing runs (when stopped) back to
the injection values."

A FRESH run already seeds these branches from the injection
(``GlobalFit.load_info``; ``{BRANCH}_START_FACTOR=0`` in the launchers gives
the exact injection) once the parent-run source warm start is off
(``SOURCE_WARM_START_STORE`` empty, the launchers' default since 2026-10-22).
A RESUMED store continues from its own last row. This module replaces that
row's coordinates, for every branch whose settings field
``reset_to_injection_on_resume`` is True (env
``{BRANCH}_RESET_TO_INJECTION_ON_RESUME=1``, armed for one launch by
``scripts/flagship/relaunch_sources_at_injection.sh``), with the injection
rows the run built for this launch: the same ``source_info[branch].injection``
a fresh start uses, so the catalogue ids, the merger-window filter and the
sampling basis are this run's own, never re-derived here.

WHAT IS REPLACED: every leaf, on every walker, at every rung -- the main
state's engine ladder (the cold chain only, in the stock variants) and the
branch sub-state's full tempered ladder -- with the EXACT injection, and the
leaves are switched on (these are fixed-leaf branches). No ``START_FACTOR``
scatter: the request is "back to the injection values", and a random draw
taken separately on each MPI rank could leave the ranks disagreeing. The
branch's per-leaf ``d_h`` / ``h_h`` records are cleared to NaN (the "not
recorded" convention; the source move refills them at its next visit), and
its entries in the eigen-table sidecar (``<store>_eigen_tables.pkl``) are
dropped -- after a backup copy -- so the first visit builds tables, step
widths and tuning at the injection, as a fresh run would, instead of
adopting the ones learned at the old positions.

WHAT IS NOT TOUCHED, and why that is safe: the per-leaf ladders, the GB /
noise branches and the band records stay as stored. The resume's
``setup_acs(rebuild_residuals=True)`` rebuilds the residual and the cold
log-likelihood from these coordinates, and the source moves recompute every
rung's log-likelihood from the current coordinates at each leaf visit
(``prev_logl`` in ``ResidualAddOneRemoveOneMove.propose_local``), so no
stored log-likelihood is trusted afterwards. The main state's cold row and
the sub-state's cold row are written from one array, so the moves' cold-row
consistency check holds.

ONE LAUNCH, REQUEUE-SAFE: a SLURM requeue re-runs the job with the
environment it was submitted with, so the knobs are still set. Each applied
reset writes ``<store>_source_reset.json`` (the job id, ``SLURM_RESTART_COUNT``,
the stored row it was applied at, the wall time, the branches). A launch of
the SAME job that is a requeue (its restart count went up, or either count is
unknown) skips the reset when the state it resumes already carries progress
made after the reset: the store has rows past the recorded one, or the state
came from a mid-iteration checkpoint holding a later row or written after the
marker. Otherwise -- nothing persisted since the reset -- it is applied again.
A new job, or the same allocation launched again without a requeue (equal
restart counts), applies it afresh.

REFUSES (``ValueError``) rather than guess: an armed branch the resumed state
does not carry, a branch with no injection, and an injection whose rows do
not match the stored ``(nleaves, ndim)`` exactly (a single row is broadcast
only to a single-leaf branch). WARNS when a knob is set for a branch the run
does not configure. The leaf ORDER is not checked: the injection rows follow
this launch's sorted id list exactly as a plain resume's per-leaf fills do,
so a reordered id list is the same hazard with or without the reset.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import typing

import numpy as np

#: The fixed-leaf source branches the reset covers (VGB, MBHB, EMRI, SOBBH).
RESET_BRANCHES = ("vgb", "mbh", "emri", "sobbh")

#: The settings field (one per branch) that arms the reset for one launch.
RESET_FIELD = "reset_to_injection_on_resume"

#: Suffix of the marker written next to the store after an applied reset.
MARKER_SUFFIX = "_source_reset.json"



def reset_knob(branch: str) -> str:
    """``{BRANCH}_RESET_TO_INJECTION_ON_RESUME``, the env knob of ``branch``."""
    return f"{str(branch).upper()}_RESET_TO_INJECTION_ON_RESUME"


def armed_branches(source_info: typing.Mapping) -> typing.List[str]:
    """The branches of :data:`RESET_BRANCHES` whose settings arm the reset.

    Args:
        source_info: ``curr.source_info`` (branch name -> settings object).

    Returns:
        list: Branch names, in :data:`RESET_BRANCHES` order.
    """
    out = []
    for b in RESET_BRANCHES:
        info = source_info.get(b) if source_info is not None else None
        if info is not None and bool(getattr(info, RESET_FIELD, False)):
            out.append(b)
    return out


def _env_true(environ, var) -> bool:
    """The settings' boolean spelling (``stock.base``): 1/true/yes/on."""
    from .stock.base import _BOOL_TRUE

    return str(environ.get(var, "")).strip().lower() in _BOOL_TRUE


def _armed_but_unconfigured(source_info, environ=None) -> typing.List[str]:
    """Knobs set in the environment for branches the run does not configure."""
    environ = os.environ if environ is None else environ
    return [b for b in RESET_BRANCHES
            if (source_info is None or source_info.get(b) is None)
            and _env_true(environ, reset_knob(b))]


def injection_rows(info, branch: str, nleaves: int, ndim: int) -> np.ndarray:
    """``info.injection`` as an ``(nleaves, ndim)`` float array, validated.

    A flat ``(ndim,)`` or single-row injection is accepted only for a
    single-leaf branch: broadcasting one row over several stored leaves would
    put the same source into every leaf.

    Raises:
        ValueError: No injection, or rows that do not match the stored leaves.
    """
    inj = getattr(info, "injection", None)
    if inj is None:
        raise ValueError(
            f"{reset_knob(branch)}=1 but the '{branch}' branch has no catalogue "
            "injection this launch (source_info.injection is None), so there is "
            "nothing to reset it to. Unset the knob for this branch.")
    inj = np.asarray(inj, dtype=float)
    if inj.ndim == 1:
        inj = inj[None, :]
    if inj.ndim != 2 or inj.shape != (int(nleaves), int(ndim)):
        raise ValueError(
            f"{reset_knob(branch)}=1 but this launch's '{branch}' injection has "
            f"shape {np.asarray(getattr(info, 'injection')).shape} while the resumed "
            f"state stores (nleaves, ndim) = ({nleaves}, {ndim}). The store was "
            "built from a different id list or sampling basis; refusing rather "
            "than put the wrong source into a leaf.")
    return np.array(inj, dtype=float, copy=True)


def reset_sources_to_injection(state, source_info, logger=None) -> dict:
    """Set every armed branch of ``state`` to its catalogue injection, in place.

    Args:
        state: The resumed ``GFState`` (main ``branches`` + ``sub_states``).
        source_info: ``curr.source_info`` (branch -> settings with
            ``injection`` and :data:`RESET_FIELD`).
        logger: Optional logger; one ``[SOURCE-RESET]`` INFO line per branch.

    Returns:
        dict: ``{branch: {"nleaves", "ndim", "nwalkers", "ntemps_main",
        "ntemps_sub", "max_abs_move_per_leaf"}}`` for the branches reset (``max_abs_move_per_leaf``: the cold chain's largest
        absolute change per leaf, over walkers and columns, in the sampling
        basis).

    Raises:
        ValueError: See the module docstring.
    """
    out = {}
    for b in armed_branches(source_info):
        branches = getattr(state, "branches", None) or {}
        if b not in branches:
            raise ValueError(
                f"{reset_knob(b)}=1 but the resumed state carries no '{b}' branch "
                f"(branches: {sorted(branches)}).")
        main = branches[b]
        nt_main, nw, nleaves, ndim = np.shape(main.coords)
        inj = injection_rows(source_info[b], b, nleaves, ndim)
        old_cold = np.array(main.coords[0], dtype=float, copy=True)
        sub = (getattr(state, "sub_states", None) or {}).get(b)
        nt_sub = 0
        if sub is not None and getattr(sub, "tempered_initialized", False):
            if tuple(np.shape(sub.coords)[1:]) != (nw, nleaves, ndim):
                raise ValueError(
                    f"{reset_knob(b)}=1 but the '{b}' sub-state's tempered block "
                    f"{np.shape(sub.coords)} does not match the main branch "
                    f"(nwalkers, nleaves, ndim) = ({nw}, {nleaves}, {ndim}).")
        main.coords[:] = inj[None, None]
        main.inds[:] = True
        if sub is not None and getattr(sub, "tempered_initialized", False):
            sub.coords[:] = inj[None, None]
            sub.inds[:] = True
            nt_sub = int(np.shape(sub.coords)[0])
            for name in ("d_h", "h_h"):
                arr = getattr(sub, name, None)
                if arr is not None:
                    arr[...] = np.nan
        moved = np.abs(old_cold - inj[None]).max(axis=(0, 2))
        out[b] = dict(nleaves=int(nleaves), ndim=int(ndim), nwalkers=int(nw),
                      ntemps_main=int(nt_main), ntemps_sub=nt_sub,
                      max_abs_move_per_leaf=moved)
        if logger is not None:
            logger.info(
                "[SOURCE-RESET] %s: %d leaves x %d walkers set to the catalogue "
                "injection (exact, no START_FACTOR scatter) at all %d rungs of "
                "the sub-state ladder and %d of the main state (%s=1); the cold "
                "chain's largest |change| per leaf in the sampling basis: %s. "
                "The residual and the log-likelihood are rebuilt from these "
                "coordinates; the per-leaf d_h / h_h records read NaN until the "
                "branch's move runs.",
                b, int(nleaves), int(nw), nt_sub, int(nt_main), reset_knob(b),
                np.array2string(moved, precision=4, separator=", "))
    return out


def reset_marker_path(store_path) -> typing.Optional[str]:
    """``<store minus .h5>_source_reset.json``, or ``None`` without a store."""
    if not store_path:
        return None
    stem = str(store_path)
    if stem.endswith(".h5"):
        stem = stem[:-3]
    return stem + MARKER_SUFFIX


def _as_int(v) -> typing.Optional[int]:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v) -> typing.Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _read_marker(path) -> typing.Optional[dict]:
    """The marker's fields, parsed; ANY problem means "no marker" (reset)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fp:
            got = json.load(fp)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(got, dict) or not got.get("job_id"):
        return None
    branches = got.get("branches")
    return dict(job_id=str(got.get("job_id")),
                restart_count=_as_int(got.get("restart_count")),
                stored_iteration=_as_int(got.get("stored_iteration")),
                unix_time=_as_float(got.get("unix_time")),
                branches=[str(b) for b in branches] if isinstance(branches, list) else [])


def _write_json_atomic(path, payload) -> bool:
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "w") as fp:
            json.dump(payload, fp, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return True
    except Exception:  # noqa: BLE001
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def drop_eigen_table_entries(store_path, branches, logger=None, tag="") -> int:
    """Remove the ``branches``' entries from the store's eigen-table sidecar.

    A backup copy ``<sidecar>.pre_source_reset<tag>`` is written first; the
    sidecar is then replaced atomically with the other branches' entries
    intact. A missing sidecar is a no-op. Never raises: a failure is logged
    and the stale tables are merely adopted (valid, just adapted to the old
    positions).

    Returns:
        int: The number of entries removed.
    """
    try:
        from .moves.eigen_table_persist import load_sidecar, sidecar_path
    except Exception:  # noqa: BLE001
        return 0
    path = sidecar_path(store_path) if store_path else None
    if not path or not os.path.exists(path):
        return 0
    payload = load_sidecar(path)
    entries = payload.get("entries", {})
    drop = [k for k in entries if str(k).split(":", 1)[0] in set(branches)]
    if not drop:
        return 0
    backup = f"{path}.pre_source_reset{tag}"
    try:
        if not os.path.exists(backup):        # keep the TRUE pre-reset copy
            shutil.copy2(path, backup)
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning(
                "[SOURCE-RESET] could not back up the eigen-table sidecar %s (%r); "
                "leaving it untouched (the old tables are adopted).", path, exc)
        return 0
    for k in drop:
        entries.pop(k, None)
    import pickle

    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        if logger is not None:
            logger.warning(
                "[SOURCE-RESET] could not rewrite the eigen-table sidecar %s (%r); "
                "the old tables are adopted.", path, exc)
        return 0
    if logger is not None:
        logger.info(
            "[SOURCE-RESET] dropped %d eigen-table sidecar entr%s for %s from %s "
            "(backup %s.pre_source_reset%s): the first visit builds them at the "
            "injection.", len(drop), "y" if len(drop) == 1 else "ies",
            ", ".join(branches), path, path, tag)
    return len(drop)


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def is_requeue_after_progress(prior, *, job_id, restart_count, stored_iteration,
                              ckpt_meta=None) -> bool:
    """Whether this launch must SKIP the reset (a requeue that resumes progress).

    Args:
        prior: :func:`_read_marker`'s dict, or ``None``.
        job_id: This launch's SLURM job id (``None`` outside SLURM).
        restart_count: This launch's ``SLURM_RESTART_COUNT`` (``None`` unset).
        stored_iteration: The store's ``iteration`` when it was opened.
        ckpt_meta: The mid-iteration checkpoint's meta when the state came
            from one (``stored_iteration``, ``unix_time``), else ``None``.

    Returns:
        bool: True = skip (same job, a requeue, and the resumed state already
        holds progress made after the reset).
    """
    if not job_id or not prior or prior.get("job_id") != str(job_id):
        return False
    rc_then, rc_now = prior.get("restart_count"), _as_int(restart_count)
    if rc_then is not None and rc_now is not None and rc_now <= rc_then:
        return False          # the same allocation launched again, not a requeue
    row = prior.get("stored_iteration")
    if row is None:
        return False
    its = [x for x in (_as_int(stored_iteration),
                       _as_int((ckpt_meta or {}).get("stored_iteration"))) if x is not None]
    if its and max(its) > row:
        return True           # a row (or a checkpoint of a later row) was saved since
    t_ckpt = _as_float((ckpt_meta or {}).get("unix_time"))
    t_reset = prior.get("unix_time")
    if t_ckpt is not None and t_reset is not None and t_ckpt > t_reset:
        return True           # a checkpoint written after the reset holds its progress
    return False


def maybe_reset_on_resume(state, source_info, *, fresh_start: bool, logger=None,
                          store_path=None, stored_iteration=None, job_id=None,
                          restart_count=None, ckpt_meta=None, environ=None) -> dict:
    """The ``load_info`` entry point: reset a LOADED state, once per job.

    Args:
        state: The state ``load_info`` is about to return.
        source_info: ``curr.source_info``.
        fresh_start: True when ``load_info`` built ``state`` from the priors /
            injection rather than loading it from a store (then nothing is
            reset; an armed knob is reported).
        logger: Optional logger.
        store_path: The run's store (``general_info.main_file_path``): the
            marker and the eigen sidecar live next to it.
        stored_iteration: The store's ``iteration`` when it was opened.
        job_id: ``SLURM_JOB_ID`` (a requeue keeps it), ``None`` outside SLURM.
        restart_count: ``SLURM_RESTART_COUNT`` (set on a requeue), or ``None``.
        ckpt_meta: The mid-iteration checkpoint's meta when ``state`` came
            from one, else ``None``.
        environ: Environment mapping for the unconfigured-branch warning
            (default ``os.environ``).

    Returns:
        dict: As :func:`reset_sources_to_injection`; ``{}`` when nothing was
        reset.
    """
    if state is None:
        return {}
    for b in _armed_but_unconfigured(source_info, environ):
        if logger is not None:
            logger.warning(
                "[SOURCE-RESET] %s=1 is set but this run configures no '%s' "
                "branch -- nothing to reset there.", reset_knob(b), b)
    armed = armed_branches(source_info)
    if not armed:
        return {}
    if fresh_start:
        if logger is not None:
            logger.info(
                "[SOURCE-RESET] fresh start: %s armed but not applied -- a fresh "
                "run seeds these branches from the injection itself "
                "({BRANCH}_START_FACTOR scatter, 0 in the launchers, unless a "
                "source warm start was armed on the line).",
                ", ".join(reset_knob(b) + "=1" for b in armed))
        return {}
    marker_path = reset_marker_path(store_path)
    prior = _read_marker(marker_path)
    if is_requeue_after_progress(prior, job_id=job_id, restart_count=restart_count,
                                 stored_iteration=stored_iteration, ckpt_meta=ckpt_meta):
        if logger is not None:
            logger.info(
                "[SOURCE-RESET] skipped: job %s (restart %s) already reset %s at "
                "stored row %s and the state it resumes carries progress made since "
                "(store iteration %s%s); the chain continues.",
                job_id, restart_count, ", ".join(prior.get("branches", [])),
                prior.get("stored_iteration"), stored_iteration,
                "" if not ckpt_meta else (
                    f", mid-iteration checkpoint row {ckpt_meta.get('stored_iteration')}"))
        return {}
    out = reset_sources_to_injection(state, source_info, logger)
    if not out:
        return out
    now = _now_utc()
    tag = f"_{job_id}" if job_id else now.strftime("_%Y%m%dT%H%M%S")
    drop_eigen_table_entries(store_path, list(out), logger, tag=tag)
    if marker_path:
        payload = dict(job_id=(str(job_id) if job_id else None),
                       restart_count=_as_int(restart_count),
                       stored_iteration=_as_int(stored_iteration),
                       unix_time=now.timestamp(),
                       written_utc=now.isoformat(timespec="seconds"),
                       branches=list(out))
        if not _write_json_atomic(marker_path, payload) and logger is not None:
            logger.warning(
                "[SOURCE-RESET] could not write %s: a SLURM requeue of this job "
                "would apply the reset again.", marker_path)
    return out
