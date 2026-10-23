"""Source warm start: MBH / EMRI / SOBBH leaves from a previous run's cold chain.

USER REQUEST 2026-10-08, for the 9mo / 1yr production runs:

    *"start the 9mo sources from the 6mo cold chain's final positions instead
    of catalogue truth ... let's do this for any source over SNR 10 at 6 mo.
    Otherwise start how we did at 6mo for sources under SNR 10 (computed at
    six months)"*

WHAT IS READ. The seed store's per-branch sub-backend group
``global_fit/sub_backend/<branch>`` at its LAST WRITTEN row (the main group's
``iteration`` attr minus one):

* ``chain`` ``(ntemps, nwalkers, nleaves, ndim)`` -- the branch's whole
  per-leaf ladder (MBH / EMRI / SOBBH are :class:`~lisatools.globalfit.state.
  PerLeafLadderState` sub-states: one rung count for the branch, one
  ``betas_all`` row per leaf);
* ``inds`` -- must be True on the cold rung (these are fixed-leaf branches;
  an all-False row is a row the saver has not written yet, see below);
* ``h_h`` ``(nwalkers, nleaves)`` -- the per-leaf COLD-CHAIN ``<h|h>`` the
  add/remove moves record at the end of every leaf visit
  (``_record_leaf_inner_products``; no rung axis). It is gated by
  ``{BRANCH}_RECORD_DH``: OFF on the slow container path, ON for the
  fast-kernel moves production runs (``MBHBatchedLikeMove``,
  ``EMRIDirectLikeMove``, ``SOBBHChunkedLikeMove`` which also serves
  ``SOBBH_LIKELIHOOD=lookup``). Off -> the stored record is NaN.

SNR. ``sqrt(h_h)`` of a cold walker is the optimal SNR of that walker's
template in the seed run's data under that walker's noise model. A leaf's
seed SNR is the MEDIAN over cold walkers: the decision is per SOURCE (all
walkers of a leaf start the same way, so the walkers cannot vote
individually), and the median is the posterior-typical value, immune to
one stuck walker. A leaf whose record is non-finite on ANY walker has an
UNKNOWN SNR: refused by default (``SOURCE_WARM_START_SNR_UNKNOWN``).

LEAF IDENTITY. Stored leaves are the seed run's catalogue injections in
SORTED catalogue-id order (``prepare_*_branch``: ``sorted(cat.keys())``,
merger-window filtered for MBH). The store does NOT record the id list, so
it comes from ``{BRANCH}_SOURCE_WARM_START_IDS`` (the seed run's
``MBHB_IDS`` / ``EMRI_IDS`` / ``SOBHB_IDS``, any order; sorted here). The
new run's leaf ids are ``injection_ids`` (recorded by ``prepare_*_branch``).
Leaves map BY CATALOGUE ID, never by leaf index; a new-run id the store
does not hold starts at the injection.

PLACEMENT. Walker ``w`` <- seed walker ``w`` (cycled ``w % n_seed`` with a
warning when the counts differ). When this run's branch ladder has the
seed's rung count the whole ladder is copied rung by rung; otherwise the
seed's cold positions go on every rung (what ``START_FACTOR=0`` does with
the injection). Every warm start is checked against THIS run's prior; a
point outside it refuses the launch. No random numbers are consumed.

WHERE IT SITS. ``GlobalFit._source_warm_start`` (``run.py``) calls
:func:`warm_start_branch` once per armed branch while a FRESH store's start
coordinates are built (``load_info``); a resume keeps its chain and never
calls it. The knobs are fields of the branch settings
(``source_runtime.py``), env-seeded:

* ``SOURCE_WARM_START_STORE`` -- the seed store (default unset = off;
  shared by the three branches). Launchers: ``${GF_SEED_STORE}``
  (explicitly empty on a run without a parent).
* ``SOURCE_WARM_START_SNR_MIN`` -- default 10.0 (launchers 10).
* ``SOURCE_WARM_START_SNR_UNKNOWN`` -- ``refuse`` (default) / ``truth`` /
  ``warm`` for a NaN ``<h|h>`` record (launchers: unset).
* ``MBH_SOURCE_WARM_START_IDS`` / ``EMRI_SOURCE_WARM_START_IDS`` /
  ``SOBBH_SOURCE_WARM_START_IDS`` -- the SEED run's id list per branch
  (:func:`parse_id_list`; ``none`` disables the branch). Launchers: the
  parent's lists (9mo: MBH ``2,5,16,18``, EMRI ``0,...,7``, SOBBH
  ``0,...,5``; 12mo: MBH ``2,5,7,12,16,18``).

Example:
    Dry run on the cluster before launching (prints each id's seed SNR and
    the warm / injection decision; ``--snr-min`` defaults to 10):

    .. code-block:: bash

        python -m lisatools.globalfit.warmstart.sources \\
            --store gf_prod_6mo_testing.h5 --branch mbh --ids 2,5,16,18
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import typing

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "SOURCE_WARM_BRANCHES",
    "SNR_UNKNOWN_CHOICES",
    "SeedLeaves",
    "SourceWarmStartError",
    "parse_id_list",
    "read_seed_leaves",
    "place_warm_leaves",
    "warm_start_branch",
    "main",
]

#: Branches the warm start serves (fixed-leaf, per-leaf-ladder sub-states).
SOURCE_WARM_BRANCHES = ("mbh", "emri", "sobbh")
#: ``SOURCE_WARM_START_SNR_UNKNOWN`` values.
SNR_UNKNOWN_CHOICES = ("refuse", "truth", "warm")
#: Top-level HDF5 group of a global-fit store.
_GROUP = "global_fit"
#: Rows the reader may step back past unwritten trailing rows.
_ROW_LOOKBACK = 3
#: Data-start agreement tolerance [s].
_DATA_T0_TOL = 1e-3


class SourceWarmStartError(ValueError):
    """A source warm start that cannot be honoured: the launch is refused."""


def parse_id_list(raw) -> typing.Optional[tuple]:
    """``"2,5,16,18"`` -> ``(2, 5, 16, 18)``; ``"none"``/``"off"`` -> ``()``.

    The ``cast`` of ``{BRANCH}_SOURCE_WARM_START_IDS`` (module level and named,
    so the settings tree pickles). ``()`` disables the branch's warm start;
    an empty env value never reaches here (``env_resolve`` treats it as unset).

    Args:
        raw (str, tuple, list or None): The id list.

    Returns:
        tuple of int or None: The ids in the given order (``None`` for
        ``None``).

    Raises:
        ValueError: On a duplicate id or a non-integer entry.

    Example:
        >>> from lisatools.globalfit.warmstart.sources import parse_id_list
        >>> parse_id_list("2,5,16,18")
        (2, 5, 16, 18)
        >>> parse_id_list("none")
        ()
    """
    if raw is None:
        return None
    if isinstance(raw, (tuple, list)):
        vals = tuple(int(x) for x in raw)
    else:
        text = str(raw).strip()
        if text.lower() in ("none", "off"):
            return ()
        vals = tuple(int(x) for x in text.split(",") if x.strip())
    if len(set(vals)) != len(vals):
        raise ValueError(f"source warm-start id list {raw!r} has duplicate ids")
    return vals


def _ids_text(ids) -> str:
    """Comma-joined ids for messages."""
    return ",".join(str(int(i)) for i in ids)


def _ids_knob(branch: str) -> str:
    """The branch's id-list env knob name, ``{BRANCH}_SOURCE_WARM_START_IDS``."""
    return f"{branch.upper()}_SOURCE_WARM_START_IDS"


@dataclasses.dataclass
class SeedLeaves:
    """One branch of a seed store at one stored row.

    Attributes:
        store: The file actually read (the primary, or its running backup).
        branch: Branch name.
        row: Stored row index read.
        ids: Catalogue ids of the stored leaves, in stored (sorted) order.
        coords: ``(ntemps, nwalkers, nleaves, ndim)`` sub-backend chain row.
        inds: ``(ntemps, nwalkers, nleaves)``.
        h_h: ``(nwalkers, nleaves)`` cold-chain ``<h|h>`` record.
        data_t0: The seed run's data start from its ``noise_model_identity``
            (``None`` when the store predates the record).
    """

    store: str
    branch: str
    row: int
    ids: tuple
    coords: np.ndarray
    inds: np.ndarray
    h_h: np.ndarray
    data_t0: typing.Optional[float] = None

    @property
    def ntemps(self) -> int:
        """Rungs of the seed branch ladder."""
        return int(self.coords.shape[0])

    @property
    def nwalkers(self) -> int:
        """Walkers of the seed store."""
        return int(self.coords.shape[1])

    def snr(self) -> np.ndarray:
        """``(nleaves,)`` median-over-cold-walkers ``sqrt(h_h)``; NaN = unknown."""
        hh = np.asarray(self.h_h, dtype=float)
        good = np.isfinite(hh) & (hh >= 0.0)
        per_walker = np.sqrt(np.where(good, hh, 0.0))
        med = np.median(per_walker, axis=0)
        return np.where(good.all(axis=0), med, np.nan)


def _row_written(inds_row) -> bool:
    """A sub-backend row the saver has written (fixed-leaf: cold inds True).

    ``GFHDFBackend.save_step_main`` advances the main ``iteration`` before the
    sub-backend rows land, and ``grow`` preallocates rows that read back as
    zeros, so a reader in between sees an all-False ``inds`` row.
    """
    return bool(np.asarray(inds_row)[0].any())


def _read_last_written(path: str, branch: str) -> typing.Optional[dict]:
    """Raw last-written-row arrays of ``branch``, or ``None`` if the store lacks it.

    Tries the last ``_ROW_LOOKBACK`` (3) rows, newest first, skipping
    unwritten ones. Returns a dict with keys ``row``, ``coords``, ``inds``,
    ``h_h`` (all NaN when the store has no record) and ``data_t0``.
    Raises ``OSError`` (h5py's error class for a
    torn/unreadable file or chunk) for the caller's backup fallback, and
    :class:`SourceWarmStartError` for a store without rows or without a
    written row in the lookback.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if _GROUP not in f:
            raise OSError(f"{path}: no '{_GROUP}' group")
        g = f[_GROUP]
        if "sub_backend" not in g or branch not in g["sub_backend"]:
            return None
        sub = g["sub_backend"][branch]
        it = int(g.attrs.get("iteration", 0))
        if it <= 0:
            raise SourceWarmStartError(f"{path}: the store holds no saved rows")
        for row in range(it - 1, max(it - 1 - _ROW_LOOKBACK, -1), -1):
            if row >= sub["inds"].shape[0]:
                continue
            inds = np.asarray(sub["inds"][row], dtype=bool)
            if not _row_written(inds):
                logger.info(
                    "[SOURCE-WARM] %s: %s row %d is not written yet (saver mid-write); "
                    "stepping back", branch, os.path.basename(path), row,
                )
                continue
            out = dict(
                row=row,
                coords=np.asarray(sub["chain"][row], dtype=float),
                inds=inds,
                h_h=(np.asarray(sub["h_h"][row], dtype=float) if "h_h" in sub
                     else np.full(inds.shape[1:], np.nan)),
            )
            ident = g.get("noise_model_identity")
            t0 = None if ident is None else ident.attrs.get("data_t0")
            out["data_t0"] = None if t0 is None else float(t0)
            return out
        raise SourceWarmStartError(
            f"{path}: no written {branch} row among the last {_ROW_LOOKBACK} "
            f"(iteration {it})"
        )


def read_seed_leaves(store: str, branch: str, ids) -> typing.Optional[SeedLeaves]:
    """The seed store's last cold-chain ladder for ``branch``, labelled by catalogue id.

    A torn / unreadable primary (a live run's saver mid-write, a preempted
    job) falls back to ``<store>_running_backup_copy.h5``.

    Args:
        store: Seed run's global-fit h5 (``GF_SEED_STORE``).
        branch: ``mbh`` / ``emri`` / ``sobbh``.
        ids (str or sequence of int): The seed run's catalogue id list for
            the branch (any order; :func:`parse_id_list` input).

    Returns:
        SeedLeaves or None: :class:`SeedLeaves` (ids sorted), or ``None``
        when the store has no such branch.

    Raises:
        SourceWarmStartError: When the store does not exist, it (and its
            backup, or it without a backup) cannot be read, it holds no
            written row, or ``ids`` does not have one id per stored leaf.
    """
    if not os.path.exists(store):
        raise SourceWarmStartError(
            f"[SOURCE-WARM] seed store {store} does not exist "
            "(SOURCE_WARM_START_STORE)"
        )
    path = store
    try:
        raw = _read_last_written(store, branch)
    except OSError as err:
        backup = store[:-3] + "_running_backup_copy.h5"
        if not os.path.exists(backup):
            raise SourceWarmStartError(
                f"[SOURCE-WARM] seed store {store} is unreadable ({err}) and has "
                "no running backup copy beside it."
            ) from err
        logger.warning(
            "[SOURCE-WARM] seed store %s unreadable (%s); reading its running "
            "backup copy %s", store, err, backup,
        )
        try:
            raw = _read_last_written(backup, branch)
        except OSError as err2:
            raise SourceWarmStartError(
                f"[SOURCE-WARM] seed store {store} AND its backup are unreadable "
                f"({err}; {err2})."
            ) from err2
        path = backup
    if raw is None:
        return None
    ids = tuple(sorted(parse_id_list(ids)))
    nleaves = raw["coords"].shape[2]
    if len(ids) != nleaves:
        raise SourceWarmStartError(
            f"[SOURCE-WARM] {branch}: {_ids_knob(branch)} names {len(ids)} ids "
            f"({_ids_text(ids)}) but the seed store's {branch} branch has "
            f"{nleaves} leaves. The list must be the SEED run's own id list "
            f"(after its merger-window filter for MBH), not this run's."
        )
    return SeedLeaves(store=path, branch=branch, ids=ids, **raw)


def _prior_failure(prior, rows, cid, branch, walkers, rungs) -> str:
    """Name the first out-of-prior warm start: id, walker, rung, parameter, value."""
    rows = np.atleast_2d(rows)
    lp = np.atleast_1d(np.asarray(prior.logpdf(rows), dtype=float))
    bad = np.flatnonzero(~np.isfinite(lp))
    i = int(bad[0])
    names = list(getattr(prior, "key_order", []) or [])
    params = []
    for inds, dist in getattr(prior, "priors", []):
        inds = np.atleast_1d(inds)
        try:
            x = rows[i, inds[0]] if len(inds) == 1 else rows[i:i + 1, inds]
            ok = np.all(np.isfinite(np.asarray(dist.logpdf(x), dtype=float)))
        except Exception:                       # noqa: BLE001 -- diagnosis only
            continue
        if not ok:
            for c in inds:
                label = names[c] if c < len(names) else f"column {c}"
                params.append(f"{label} = {rows[i, c]:.6g}")
    what = ", ".join(params) if params else "a joint/domain constraint of the prior"
    return (
        f"{branch} id {cid}: walker {walkers[i]} rung {rungs[i]}: {what} "
        f"(point {np.array2string(rows[i], precision=6)})"
    )


def place_warm_leaves(
    branch: str,
    coords,
    *,
    new_ids,
    seed: SeedLeaves,
    snr_min: float,
    ntemps_branch: int,
    snr_unknown: str = "refuse",
    prior=None,
) -> typing.Tuple[np.ndarray, str]:
    """Overwrite the warm leaves of an injection-seeded start block.

    Args:
        branch: Branch name (for messages).
        coords (np.ndarray): ``(nt_draw, nwalkers, nleaves, ndim)`` start
            block exactly as today's ``START_FACTOR`` path built it; returned
            leaves that are not warm are byte-identical to it.
        new_ids (sequence of int): This run's catalogue ids in leaf order
            (``injection_ids``).
        seed: :func:`read_seed_leaves` output.
        snr_min: Warm iff the seed SNR is STRICTLY above this.
        ntemps_branch: This run's ladder size for the branch.
        snr_unknown: ``refuse`` / ``truth`` / ``warm`` for a NaN record.
        prior (ProbDistContainer, optional): The branch's prior container
            (``logpdf``); every warm start on the branch ladder must be
            inside it. Default ``None`` = no check.

    Returns:
        tuple[np.ndarray, str]: ``(coords, line)`` -- a new float array of
        the input's shape and the one-line ``[SOURCE-WARM]`` summary.

    Raises:
        ValueError: For an unknown ``snr_unknown``.
        SourceWarmStartError: On an id / leaf-count or ndim mismatch, an
            unknown SNR under ``refuse``, a dead / non-finite seed leaf, or a
            warm start outside ``prior``.
    """
    if snr_unknown not in SNR_UNKNOWN_CHOICES:
        raise ValueError(
            f"SOURCE_WARM_START_SNR_UNKNOWN must be one of {SNR_UNKNOWN_CHOICES}; "
            f"got {snr_unknown!r}"
        )
    out = np.array(coords, dtype=float, copy=True)
    nt_draw, nw, nl, nd = out.shape
    new_ids = tuple(int(i) for i in new_ids)
    if len(new_ids) != nl:
        raise SourceWarmStartError(
            f"[SOURCE-WARM] {branch}: {len(new_ids)} injection_ids for {nl} leaves"
        )
    if seed.coords.shape[-1] != nd:
        raise SourceWarmStartError(
            f"[SOURCE-WARM] {branch}: the seed store samples {seed.coords.shape[-1]} "
            f"parameters, this run {nd}"
        )
    nt_b = min(int(ntemps_branch), nt_draw)
    where = {cid: k for k, cid in enumerate(seed.ids)}
    snrs = seed.snr()

    warm, below, absent, unknown_truth, refused = [], [], [], [], []
    for j, cid in enumerate(new_ids):
        k = where.get(cid)
        if k is None:
            absent.append(cid)
            continue
        s = float(snrs[k])
        if not np.isfinite(s):
            if snr_unknown == "refuse":
                refused.append(cid)
            elif snr_unknown == "truth":
                unknown_truth.append(cid)
            else:
                warm.append((j, k, cid, s))
        elif s > float(snr_min):
            warm.append((j, k, cid, s))
        else:
            below.append((cid, s))
    if refused:
        raise SourceWarmStartError(
            f"[SOURCE-WARM] {branch}: seed SNR UNKNOWN for id(s) {_ids_text(refused)} "
            f"-- {os.path.basename(seed.store)} row {seed.row} holds a non-finite "
            f"cold-chain <h|h> record (the run had {branch.upper()}_RECORD_DH off, or "
            f"the move never recorded). Refusing rather than silently starting at the "
            f"injection. Override: SOURCE_WARM_START_SNR_UNKNOWN=truth (injection "
            f"start) or =warm (seed start regardless of SNR); or "
            f"{_ids_knob(branch)}=none to drop the branch's warm start."
        )

    if nw == seed.nwalkers:
        wmap, wtext = np.arange(nw), "walker w->w"
    else:
        wmap, wtext = np.arange(nw) % seed.nwalkers, f"walker w->w%{seed.nwalkers}"
        if warm:
            logger.warning(
                "[SOURCE-WARM] %s: this run has %d walkers, the seed store %d; "
                "cycling through the seed walkers (walker w <- seed walker w %% %d).",
                branch, nw, seed.nwalkers, seed.nwalkers,
            )
    copy_ladder = seed.ntemps == int(ntemps_branch)
    rtext = (
        f"ladder copied {seed.ntemps}/{seed.ntemps} rungs" if copy_ladder
        else f"cold positions on every rung (seed {seed.ntemps} rungs, this run "
             f"{int(ntemps_branch)})"
    )

    for j, k, cid, _s in warm:
        cold_alive = np.asarray(seed.inds[0][wmap, k], dtype=bool)
        block = seed.coords[:, wmap, k] if copy_ladder else seed.coords[:1, wmap, k]
        if not cold_alive.all() or not np.all(np.isfinite(block)):
            raise SourceWarmStartError(
                f"[SOURCE-WARM] {branch} id {cid}: the seed store's leaf is dead or "
                f"non-finite at row {seed.row}; cannot start from it."
            )
        out[:, :, j] = seed.coords[0][wmap, k][None]
        if copy_ladder:
            out[:nt_b, :, j] = seed.coords[:nt_b, wmap, k]

    if prior is not None and warm:
        for j, _k, cid, _s in warm:
            rows = out[:nt_b, :, j].reshape(-1, nd)
            lp = np.atleast_1d(np.asarray(prior.logpdf(rows), dtype=float))
            if not np.all(np.isfinite(lp)):
                rungs = np.repeat(np.arange(nt_b), nw)
                walkers = np.tile(np.arange(nw), nt_b)
                raise SourceWarmStartError(
                    f"[SOURCE-WARM] warm start outside THIS run's prior -- "
                    f"{_prior_failure(prior, rows, cid, branch, walkers, rungs)}. "
                    f"Every walker and rung must start inside the prior (log_prior "
                    f"-inf freezes the sampler). Check the prior box against the seed "
                    f"run's (e.g. the MBH t_plunge window, MBH_MERGER_TIME_BUFFER), or "
                    f"start this branch at the injection with {_ids_knob(branch)}=none."
                )

    parts = []
    if warm:
        parts.append(
            f"ids {_ids_text(c for _j, _k, c, _s in warm)} from the seed cold chain "
            f"(SNR {'/'.join(f'{s:.1f}' for *_x, s in warm)}; "
            f"{os.path.basename(seed.store)} row {seed.row}, {rtext}, {wtext})"
        )
    if below:
        parts.append(
            f"ids {_ids_text(c for c, _s in below)} at the injection (seed SNR "
            f"{'/'.join(f'{s:.1f}' for _c, s in below)} <= {float(snr_min):g})"
        )
    if unknown_truth:
        parts.append(
            f"ids {_ids_text(unknown_truth)} at the injection (seed SNR unknown; "
            "SOURCE_WARM_START_SNR_UNKNOWN=truth)"
        )
    if absent:
        parts.append(f"ids {_ids_text(absent)} at the injection (not in the seed store)")
    return out, f"[SOURCE-WARM] {branch}: " + "; ".join(parts or ["no leaves"])


def warm_start_branch(
    branch: str,
    coords,
    *,
    store: str,
    seed_ids,
    new_ids,
    snr_min: float,
    ntemps_branch: int,
    snr_unknown: str = "refuse",
    prior=None,
    data_t0: typing.Optional[float] = None,
) -> typing.Tuple[np.ndarray, str]:
    """Read the seed store and place this branch's warm leaves (see the module doc).

    ``seed_ids=()`` (``{BRANCH}_SOURCE_WARM_START_IDS=none``) disables the
    branch; a store without the branch starts every leaf at the injection.
    Called by ``GlobalFit._source_warm_start`` on a fresh start only.

    Args:
        branch: ``mbh`` / ``emri`` / ``sobbh``.
        coords (np.ndarray): ``(nt_draw, nwalkers, nleaves, ndim)`` start
            block built by the ``START_FACTOR`` path.
        store: The seed store (``SOURCE_WARM_START_STORE``).
        seed_ids (sequence of int or None): The seed run's id list
            (``{BRANCH}_SOURCE_WARM_START_IDS``); ``()`` disables; ``None``
            is refused when the store holds the branch.
        new_ids (sequence of int): This run's ids in leaf order
            (``injection_ids``).
        snr_min: Warm iff the seed SNR is strictly above this
            (``SOURCE_WARM_START_SNR_MIN``).
        ntemps_branch: This run's ladder size for the branch.
        snr_unknown: ``SOURCE_WARM_START_SNR_UNKNOWN``. Default ``"refuse"``.
        prior (ProbDistContainer, optional): The branch's prior container.
            Default ``None`` (no prior check).
        data_t0: This run's data start [s]; must match the seed store's
            within 1 ms (a seed store without the record only warns).
            Default ``None`` = no check.

    Returns:
        tuple[np.ndarray, str]: ``(coords, one-line summary)``; ``coords``
        is the input object itself when the branch is disabled or absent
        from the seed store, else the new array of
        :func:`place_warm_leaves`.

    Raises:
        SourceWarmStartError: On anything it cannot honour (missing store,
            unknown ``new_ids``, ids knob unset for a store holding the
            branch, data-start mismatch, and every refusal of
            :func:`read_seed_leaves` / :func:`place_warm_leaves`).
    """
    if seed_ids is not None and len(tuple(seed_ids)) == 0:
        return coords, (
            f"[SOURCE-WARM] {branch}: disabled ({_ids_knob(branch)}=none); every "
            "leaf at the injection"
        )
    if new_ids is None:
        raise SourceWarmStartError(
            f"[SOURCE-WARM] {branch}: this run's leaf catalogue ids are unknown "
            "(injection_ids is None -- the branch injection was not built by "
            f"prepare_{branch}_branch), so leaves cannot be matched by id."
        )
    if not os.path.exists(store):
        raise SourceWarmStartError(
            f"[SOURCE-WARM] seed store {store} does not exist (SOURCE_WARM_START_STORE)"
        )
    if seed_ids is None:
        import h5py

        try:
            with h5py.File(store, "r") as f:
                has = f"{_GROUP}/sub_backend/{branch}" in f
        except OSError:
            has = True                     # let the reader's fallback decide
        if has:
            raise SourceWarmStartError(
                f"[SOURCE-WARM] {branch}: SOURCE_WARM_START_STORE is set but "
                f"{_ids_knob(branch)} is not. The store does not record which "
                "catalogue id each leaf carries: set it to the SEED run's id list "
                f"(e.g. its MBHB_IDS / EMRI_IDS / SOBHB_IDS), or =none to start "
                "this branch at the injection."
            )
        seed = None
    else:
        seed = read_seed_leaves(store, branch, seed_ids)
    if seed is None:
        return coords, (
            f"[SOURCE-WARM] {branch}: ids {_ids_text(new_ids)} at the injection "
            f"(branch not in the seed store {os.path.basename(store)})"
        )
    if data_t0 is not None:
        if seed.data_t0 is None:
            logger.warning(
                "[SOURCE-WARM] %s: %s records no data start (noise_model_identity); "
                "cannot verify it matches this run's data_t0 %.6f.",
                branch, os.path.basename(seed.store), float(data_t0),
            )
        elif abs(float(seed.data_t0) - float(data_t0)) > _DATA_T0_TOL:
            raise SourceWarmStartError(
                f"[SOURCE-WARM] {branch}: the seed store's data start "
                f"{seed.data_t0:.6f} differs from this run's {float(data_t0):.6f}. "
                "Start coordinates (EMRI p0/phases, SOBBH f_low/phi0) are referenced "
                "to the data start, so they do not carry over."
            )
    return place_warm_leaves(
        branch, coords, new_ids=new_ids, seed=seed, snr_min=snr_min,
        ntemps_branch=ntemps_branch, snr_unknown=snr_unknown, prior=prior,
    )


def main(argv=None) -> int:
    """Dry run: print a seed store's per-leaf SNRs and the warm/injection decision.

    The argparse description is the module docstring's first paragraph.
    Options: ``--store`` (seed h5), ``--branch`` (``mbh`` / ``emri`` /
    ``sobbh``), ``--ids`` (the SEED run's id list), ``--snr-min`` (default
    10.0).

    Args:
        argv (list of str, optional): Arguments; ``None`` = ``sys.argv[1:]``.

    Returns:
        int: 0, or 1 when the store has no such branch.
    """
    ap = argparse.ArgumentParser(
        prog="python -m lisatools.globalfit.warmstart.sources",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("--store", required=True, help="seed run's global-fit h5")
    ap.add_argument("--branch", required=True, choices=SOURCE_WARM_BRANCHES)
    ap.add_argument("--ids", required=True,
                    help="the SEED run's catalogue id list for the branch")
    ap.add_argument("--snr-min", type=float, default=10.0)
    args = ap.parse_args(argv)
    seed = read_seed_leaves(args.store, args.branch, args.ids)
    if seed is None:
        print(f"[SOURCE-WARM] {args.branch}: not in {args.store}")
        return 1
    hh = np.asarray(seed.h_h, dtype=float)
    print(f"[SOURCE-WARM] {args.branch}: {seed.store} row {seed.row}, "
          f"{seed.ntemps} rungs x {seed.nwalkers} walkers, data_t0 {seed.data_t0}")
    for k, (cid, s) in enumerate(zip(seed.ids, seed.snr())):
        per_w = "/".join(f"{np.sqrt(v):.1f}" if np.isfinite(v) and v >= 0 else "nan"
                         for v in hh[:, k])
        verdict = ("UNKNOWN (refused by default)" if not np.isfinite(s)
                   else "warm" if s > args.snr_min else "injection")
        print(f"  id {cid:>4}: SNR median {s:9.2f}  per walker {per_w}  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
