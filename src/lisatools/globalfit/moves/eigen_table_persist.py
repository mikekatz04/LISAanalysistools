# -*- coding: utf-8 -*-
"""Sidecar persistence for the per-leaf eigen proposal tables.

The ``(axes, sigmas)`` tables that
:meth:`~lisatools.globalfit.moves.addremovemove.ResidualAddOneRemoveOneMove.refresh_inner_move_tables`
feeds to the eryn :class:`~eryn.moves.EigenAxisMove` inner moves are
DERIVED products of an information matrix: minutes per leaf to build
(EMRI/MBH one Gram-form matrix per leaf; SOBBH a whole
``ntemps*nwalkers`` batch), kilobytes to store. Until this module they
lived only on the move object, so every process restart — the campaign
runs on a preemptible partition — repaid the full first-visit build for
every leaf of every branch.

This module writes the derived tables (plus the per-leaf visit counter
that drives the refresh cadence) to ONE pickle sidecar next to the run's
store, ``<store minus .h5>_eigen_tables.pkl``, in the spirit of the
mid-iteration checkpoint that sits in the same directory. Writes are
atomic (tmp + :func:`os.replace`) and a resume adopts an entry only when
the guards that would make the table the WRONG SHAPE or the wrong physics
agree: ``ndim``, ``ntemps``, ``nwalkers``, scope and — when both sides
know it — the data identity.

Staleness is deliberately tolerated. The expansion point an entry was
built at is recorded for diagnostics but is NOT a guard: the tables were
already frozen between refreshes (that freeze is what keeps the inner
move's ``factors == 0`` honest), and a frozen symmetric proposal is
correct Metropolis-Hastings whatever point its curvature came from —
a stale one adapts worse, nothing more. Persistence therefore changes
NOTHING about detailed balance; it only replaces recomputation at process
boundaries. The visit counter riding along means the normal cadence
refreshes an adopted table on schedule anyway.

Nothing here may ever break a run over its own cache: a missing,
truncated, corrupt or unreadable sidecar warns (once per path per reason)
and behaves exactly like an empty cache. ``EIGEN_TABLES_PERSIST=0``
restores the in-memory-only behavior.
"""

import os
import pickle
import time

import numpy as np

from .eigen_refresh import logger

__all__ = [
    "SIDECAR_SUFFIX",
    "persist_enabled",
    "sidecar_path",
    "make_entry",
    "load_sidecar",
    "load_entry",
    "save_entry",
]

#: Appended to the store path (minus ``.h5``) to name the sidecar.
SIDECAR_SUFFIX = "_eigen_tables.pkl"

#: Payload format stamp — bump if the entry layout changes incompatibly
#: (an unknown format is ignored like a corrupt file, never adopted).
#: 2 (2026-09-16): tables built before the ``prior_box_widths`` fix were
#: whitened and capped with UNIT prior widths on every string-keyed branch
#: (mbh / emri / sobbh); the adoption guards (ndim, ntemps, nwalkers, scope,
#: data identity) cannot tell such a table from a current one, so the bump
#: is what makes a warm restart rebuild them instead of proposing with the
#: old axes for a whole refresh cadence.
#: 3 (2026-09-16): the MBH ``Q`` prior became the linear-column log-uniform
#: on [1, 10] (``LogUniformLinear``) — its prior-box width went 2.30 -> 9.0,
#: so every MBH table built before it whitens the wrong box; same remedy.
FORMAT_VERSION = 3

# (path, reason) pairs already warned about, so a sampler that touches a
# broken sidecar every leaf visit logs once, not thousands of times.
_WARNED = set()


def _warn_once(path, reason, *args):
    key = (str(path), reason)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(*args)


def persist_enabled():
    """``EIGEN_TABLES_PERSIST`` env > on. Read at use, never at import."""
    val = os.environ.get("EIGEN_TABLES_PERSIST", "1")
    return str(val).strip().lower() not in ("0", "false", "no", "off")


def sidecar_path(store_path):
    """``<store minus .h5>_eigen_tables.pkl``, or ``None`` w/o a store.

    Derived exactly like
    :func:`lisatools.globalfit.midit_checkpoint.checkpoint_path` (same
    ``splitext`` of the same ``general_info.main_file_path``), so the
    eigen sidecar lands beside the midit pkl and the store's own backup.
    """
    if not store_path:
        return None
    base, _ = os.path.splitext(str(store_path))
    return base + SIDECAR_SUFFIX


def entry_key(branch, leaf):
    """One entry per (branch, leaf) — the scope of a single table build."""
    return f"{branch}:{int(leaf)}"


def make_entry(axes, sigmas, visits, *, scope, ndim, ntemps, nwalkers,
               x0=None, data_identity=None, store_iteration=None,
               sigmas_fisher=None, axis_tune=None):
    """One sidecar entry: the table as the move consumes it + guards.

    ``axes``/``sigmas`` are stored in the EXACT shape the move holds in
    ``_eigen_tables`` (the ``(ntemps, nwalkers, ndim, ndim)`` per-walker
    stash, or a single shared table), so a resume can drop them straight
    back in. ``visits`` is the post-increment per-leaf visit counter, so
    the refresh cadence continues where the killed process left off.
    ``x0`` (the expansion point) and the stamps are diagnostics only.

    ``sigmas_fisher`` (``{BRANCH}_EIGEN_AXIS_ADAPT`` on, 2026-10-14): the
    information-matrix sigmas the adapted ``sigmas`` were widened from --
    the adaptation's floor. The key is written ONLY when given, so an entry
    written with the adaptation off is the pre-adaptation entry exactly; an
    entry without it reads as "the table is the information-matrix one".
    Adding the key needs no format bump: older readers ignore it (and adopt
    the adapted sigmas, a stale-but-valid frozen proposal).

    ``axis_tune`` (``{BRANCH}_EIGEN_AXIS_ACC_TUNE`` on, 2026-10-18): the
    acceptance-driven controller's per-axis multiplier ``(ndim,)`` on the
    table's (adapted) sigma -- ``sigmas`` is already the tuned step that ran.
    Written only when given, same no-bump argument.
    """
    entry = {
        "axes": np.asarray(axes),
        "sigmas": np.asarray(sigmas),
        "visits": int(visits),
        "scope": None if scope is None else str(scope),
        "ndim": int(ndim),
        "ntemps": int(ntemps),
        "nwalkers": int(nwalkers),
        "x0": None if x0 is None else np.asarray(x0),
        "data_identity": (
            None if data_identity is None else float(data_identity)
        ),
        "store_iteration": (
            None if store_iteration is None else int(store_iteration)
        ),
        "written_at": time.time(),
    }
    if sigmas_fisher is not None:
        entry["sigmas_fisher"] = np.asarray(sigmas_fisher)
    if axis_tune is not None:
        entry["axis_tune"] = np.asarray(axis_tune, dtype=float)
    return entry


def _empty_payload():
    return {"format": FORMAT_VERSION, "entries": {}}


def load_sidecar(path):
    """Read the whole sidecar; ANY problem yields an empty payload.

    Missing file is the normal cold-start case and is silent. Truncated,
    corrupt, unpickleable or structurally wrong contents warn once and
    are then treated as an empty cache (the next save overwrites them).
    """
    if not path or not os.path.exists(path):
        return _empty_payload()
    try:
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
    except Exception as exc:  # truncated / garbage / unpickleable class
        _warn_once(
            path, "unreadable",
            "[eigen_refresh] eigen-table sidecar %s unreadable (%r); "
            "ignoring it and rebuilding the tables", path, exc,
        )
        return _empty_payload()
    if (not isinstance(payload, dict)
            or not isinstance(payload.get("entries"), dict)
            or payload.get("format") != FORMAT_VERSION):
        _warn_once(
            path, "malformed",
            "[eigen_refresh] eigen-table sidecar %s has an unexpected "
            "layout (format %r); ignoring it and rebuilding the tables",
            path, (payload.get("format")
                   if isinstance(payload, dict) else type(payload).__name__),
        )
        return _empty_payload()
    return payload


def _guards_ok(entry, key, *, scope, ndim, ntemps, nwalkers,
               data_identity=None):
    """STRICT guards: only what makes the table wrong, not merely stale.

    Shapes (square axes of the right ``ndim``, ``sigmas`` matching, a
    per-(temp, walker) stash whose leading dims are the current ladder),
    the metadata ``ndim``/``ntemps``/``nwalkers``/scope, and the data
    identity when BOTH sides know it. A mismatch logs and discards the
    entry so the caller rebuilds.
    """
    def _bad(msg, *args):
        logger.warning(
            "[eigen_refresh] discarding persisted eigen table %s: " + msg,
            key, *args,
        )
        return False

    axes = entry.get("axes")
    sigmas = entry.get("sigmas")
    if not isinstance(axes, np.ndarray) or not isinstance(sigmas, np.ndarray):
        return _bad("entry carries no arrays")
    if axes.ndim < 2 or axes.shape[-1] != axes.shape[-2]:
        return _bad("axes shape %s is not square", axes.shape)
    if axes.shape[-1] != int(ndim):
        return _bad(
            "axes ndim %d != branch ndim %d", axes.shape[-1], int(ndim)
        )
    if sigmas.shape != axes.shape[:-1]:
        return _bad(
            "sigmas shape %s does not match axes shape %s",
            sigmas.shape, axes.shape,
        )
    fisher = entry.get("sigmas_fisher")
    if fisher is not None and (
        not isinstance(fisher, np.ndarray) or fisher.shape != sigmas.shape
    ):
        return _bad(
            "sigmas_fisher shape %s does not match sigmas shape %s",
            getattr(fisher, "shape", type(fisher).__name__), sigmas.shape,
        )
    if int(entry.get("ndim", -1)) != int(ndim):
        return _bad("stored ndim %r != %d", entry.get("ndim"), int(ndim))
    # The ladder / walker-count guards apply ONLY to a per-(temp, walker)
    # stash (axes.ndim >= 4), whose rows are addressed by (temp, walker). A
    # walker-independent table -- ``walker_max`` scope, or the shared
    # fallback a ``per_walker`` branch persists -- is ONE matrix for every
    # point, so neither the walker count nor the ladder size makes it wrong:
    # the multi-rank walker-block layout builds every rank at its OWN block
    # width (B < nwalkers) and a 2-GPU -> 4-GPU continuation changes B again;
    # before 2026-09-16 that discarded all 18 MBH/EMRI/SOBBH tables and paid
    # ~45 min of information-matrix rebuilds per resume (user ruling: keep
    # them; MBH/EMRI always sit at the injection, SOBBH moved to walker_max).
    _stash = axes.ndim >= 4
    if _stash and int(entry.get("ntemps", -1)) != int(ntemps):
        return _bad(
            "stored ntemps %r != %d (temperature ladder changed)",
            entry.get("ntemps"), int(ntemps),
        )
    if _stash and int(entry.get("nwalkers", -1)) != int(nwalkers):
        return _bad(
            "stored nwalkers %r != %d", entry.get("nwalkers"), int(nwalkers)
        )
    if str(entry.get("scope")) != str(scope):
        return _bad(
            "stored scope %r != %r", entry.get("scope"), scope
        )
    # a per-(temp, walker) stash must line up with the CURRENT ladder /
    # walker count row for row — the seam slices it with the proposal mask
    if axes.ndim >= 4 and axes.shape[:2] != (int(ntemps), int(nwalkers)):
        return _bad(
            "stashed table leading dims %s != (ntemps, nwalkers) = %s",
            axes.shape[:2], (int(ntemps), int(nwalkers)),
        )
    stored_id = entry.get("data_identity")
    if stored_id is not None and data_identity is not None:
        if not np.isclose(float(stored_id), float(data_identity),
                          rtol=1e-12, atol=0.0):
            return _bad(
                "data identity %r != %r (different data/Tobs)",
                stored_id, data_identity,
            )
    return True


def load_entry(path, branch, leaf, *, scope, ndim, ntemps, nwalkers,
               data_identity=None, return_fisher=False, return_tune=False):
    """``(axes, sigmas, visits)`` for one (branch, leaf), or ``None``.

    ``None`` means "build it" — no sidecar, no entry for this leaf, or
    an entry the guards rejected (that one is logged).

    ``return_fisher=True`` appends the entry's ``sigmas_fisher`` (the
    information-matrix floor of an adapted table; ``None`` for an entry
    written with the step-scale adaptation off): ``(axes, sigmas, visits,
    sigmas_fisher)``. ``return_tune=True`` appends the entry's
    ``axis_tune`` multiplier after that (``None`` when absent or not a
    finite positive ``(ndim,)`` vector).
    """
    payload = load_sidecar(path)
    key = entry_key(branch, leaf)
    entry = payload["entries"].get(key)
    if entry is None or not isinstance(entry, dict):
        return None
    if not _guards_ok(entry, key, scope=scope, ndim=ndim, ntemps=ntemps,
                      nwalkers=nwalkers, data_identity=data_identity):
        return None
    out = (entry["axes"], entry["sigmas"], int(entry.get("visits", 0)))
    if return_fisher:
        out = out + (entry.get("sigmas_fisher"),)
    if return_tune:
        tune = entry.get("axis_tune")
        if tune is not None:
            tune = np.asarray(tune, dtype=float)
            if (tune.shape != (int(ndim),) or not np.all(np.isfinite(tune))
                    or not np.all(tune > 0)):
                tune = None
        out = out + (tune,)
    return out


def save_entry(path, branch, leaf, entry):
    """Write-through one entry, atomically. ``True`` if it landed.

    Read-modify-write of the whole payload (a handful of KB-scale
    entries): the other leaves/branches already in the file survive.
    The temp file is removed on any failure so a full disk cannot leave
    litter next to the store, and the previous sidecar stays intact —
    the cache is never worth failing a run over.
    """
    if not path:
        return False
    payload = load_sidecar(path)
    payload["entries"][entry_key(branch, leaf)] = entry
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logger.warning(
            "[eigen_refresh] could not write the eigen-table sidecar %s "
            "(%r); keeping the previous one", path, exc,
        )
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False
