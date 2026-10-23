"""Snapshot tar production, importable from the installed package.

This is ``scripts/fstat_proposal/make_snapshots.sh`` expressed in Python
so the saver rank can build a snapshot without a source checkout, a shell,
or a ``cd`` to the repo root (the shell script does all three).

THE SHELL SCRIPT IS DELIBERATELY LEFT ALONE (user instruction 2026-09-26:
"try not to touch any other code"). So yes, there are two implementations
of this recipe for now. That is a known debt, written down here rather
than discovered later: if you change the exclusion list, change it in
BOTH, and the test ``SnapshotMatchesTheShellRecipeTest`` compares the two
so the pair cannot drift silently.

The recipe, unchanged:

  1. reduce the live store with :func:`._store_extract.extract` (the big
     per-iteration chains keep only the last ``keep`` rows, cold chains
     ``cold_keep``), writing ``<store>_extract.h5``;
  2. take every file in the run dir EXCEPT the full h5 stores, backups,
     dissect dumps, rendered diagnostics, the fstat grid payloads, the
     midit checkpoint, and any prior archive;
  3. above ``log_cap_mb`` replace the DEBUG run log with a grep-filtered
     full-history file plus a raw tail (it grows ~0.5 MB/iteration and is
     the one file that outgrows the budget);
  4. ``tar -czf <run_dir>_snapshot.tar.gz``.

``output_data=True`` (CLI ``--output-data``, 2026-10-20) is the third
recipe and has no shell twin: ``<run_dir>_output_data.tar.gz`` holds the
store extract, the residual snapshot and the run folder's WDM lookup
tables (``wdm_lookup_*.h5``) and nothing else -- no logs, no page, no truth
set, no warm start. It is what a postprocessing reader needs (LATW
tutorial 09): every stored row of every cold chain (the posterior samples;
the GB cold chain is ~1 MB per row compressed, ~1.6 GB for 1685 rows of the
9-month run), the full history of the small arrays, the residual, and the
tables the EMRI and SOBBH templates are built from. The large tempered
arrays (the all-rung GB chain is ~12 MB per row) keep only the rows from
the snapshot's ``store_row`` to the store's last row: the snapshot is
written every ``GF_RESIDUAL_SNAPSHOT_EVERY`` rows and can sit a few rows
behind the last one, so a one-row window (``short``) would drop the very
row the residual belongs to. The builder ships a copy of the snapshot only
once its log likelihoods equal the stored row it names (waiting for the run
to store that row or write a new snapshot), and pins the extract to the rows
stored at that moment, so the tar stays consistent while the run saves rows
and rewrites the snapshot during the extract. Its extract is written
at gzip level 9 with HDF5's byte-shuffle filter (both built into HDF5):
~9 % smaller than the level-4 extract of the other recipes on the float
chains; LZMA at its maximum gains only ~2 % more on the same data.
"""

from __future__ import annotations

import os
import re
import tarfile
import time
from logging import getLogger
from typing import Optional

logger = getLogger(__name__)

__all__ = ["build_snapshot", "LOG_KEEP_PATTERN"]

#: Line families the filtered run log keeps. Byte-identical to the
#: ``grep -aE`` alternation in make_snapshots.sh -- the test compares them.
LOG_KEEP_PATTERN = (
    r"\[SAVE\]|\[GB_ACCEPT|\[GB_OBS_BASIS|\[GB_INMODEL_TRACE|MISMATCH|"
    r"\[GB_TEMPER_CHECK|\[GB_CAP|\[GB_ORTHO|\[stageB\]|epoch .* done in|"
    r"\[FSTAT_CTR|\[GB_TIMING|\[GB_BAND_SHUTOFF|\[GB_BAND_REVIVE|"
    r"\[V8-NOISE|\[COARSE_AUDIT|\[GB_TRUST|\[GB_VERT|\[unequal-arm|"
    r"\[galfor-modulation|\[LADDER|\[MIDIT_CKPT|WARNING|ERROR|CRITICAL|"
    r"Traceback|ll AUDIT|DELTA-vs-DELTA|sig-het engine resolved|"
    r"\[GB_SWEEP|\[LOGMIRROR|\[GB_REPLACE"
)

#: Name fragments that never enter a snapshot, whatever else matches.
_EXCLUDE_PATH_PARTS = ("fstat_grid_parts", "/dissect/", "_artifacts/diagnostics/")
_EXCLUDE_SUFFIXES = (".bak", ".tmp", ".tar.gz", ".zip",
                     "_midit_checkpoint.pkl")


def _live_store(run_dir: str) -> Optional[str]:
    """Newest ``*testing*.h5`` that is not a backup / corrupt / extract.

    Same selection as the shell's ``ls -t | grep -v``, and the same
    selection the monitor page makes -- a snapshot whose extract came from
    a different store than the page describes would be quietly incoherent.
    """
    cands = []
    for fn in os.listdir(run_dir):
        if not fn.endswith(".h5") or "testing" not in fn:
            continue
        if any(t in fn for t in ("backup", "CORRUPT", "_extract")):
            continue
        p = os.path.join(run_dir, fn)
        cands.append((os.path.getmtime(p), p))
    return max(cands)[1] if cands else None


def _reduce_log(path: str, tail_bytes: int) -> bool:
    """Write ``<base>_filtered.log`` + ``<base>_tail.log`` beside ``path``.

    True if both were written and the raw file can be skipped.
    """
    keep = re.compile(LOG_KEEP_PATTERN)
    base = path[:-4] if path.endswith(".log") else path
    try:
        with open(path, "rb") as fh, open(base + "_filtered.log", "wb") as out:
            for line in fh:
                if keep.search(line.decode("utf-8", "replace")):
                    out.write(line)
        with open(path, "rb") as fh:
            fh.seek(max(0, os.path.getsize(path) - tail_bytes))
            with open(base + "_tail.log", "wb") as out:
                out.write(fh.read())
    except Exception as e:                       # noqa: BLE001
        logger.warning("log filtering failed for %s (%s: %s); shipping it "
                       "whole", path, type(e).__name__, e)
        return False
    return True


def _filter_run_log(run_dir: str, log_cap_mb: int):
    """Above the cap, write the filtered + tail pair. Returns the raw log
    to EXCLUDE from the archive, or ``None`` to ship it whole."""
    import glob as _glob

    hits = _glob.glob(os.path.join(run_dir, "*_artifacts", "globalfit_run.log"))
    if not hits:
        return None
    runlog = hits[0]
    mb = os.path.getsize(runlog) // 1048576
    if mb <= log_cap_mb:
        return None
    logger.info("run log %d MB > %d MB -- shipping filtered + tail",
                mb, log_cap_mb)
    return runlog if _reduce_log(runlog, 20_000_000) else None


def _filter_all_logs(run_dir: str, cap_mb: int, tail_mb: int):
    """Short mode: reduce EVERY oversized ``*.log``, not just the run log.

    ⚠ THE SLURM LOGS ARE THE BULK. ``_filter_run_log`` only ever looked
    at ``*_artifacts/globalfit_run.log``; the per-job
    ``slurm_stdout_<id>.log`` files sit in the run directory root and
    shipped WHOLE. On 6mo v9 that was a single 386 MB member, which is
    most of why a full snapshot is ~330 MB compressed. A short snapshot
    is supposed to be the log information, not the log volume, so each
    oversized log becomes its filtered + tail pair.
    """
    out = set()
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            if not fn.endswith(".log"):
                continue
            if fn.endswith(("_filtered.log", "_tail.log")):
                continue          # our own products from a previous run
            p = os.path.join(root, fn)
            try:
                mb = os.path.getsize(p) // 1048576
            except OSError:
                continue
            if mb <= cap_mb:
                continue
            if _reduce_log(p, int(tail_mb) * 1_000_000):
                out.add(os.path.abspath(p))
    if out:
        logger.info("short snapshot: reduced %d oversized log(s) to "
                    "filtered + %d MB tail", len(out), tail_mb)
    return out


#: Raw bytes of stdout a 6mo job writes per SAVED iteration (measured
#: 2026-09-28). Used only to decide whether the newest job is too young
#: to carry a full iteration, in which case the previous job's stdout
#: rides along so the short tar always spans at least one.
_STDOUT_BYTES_PER_ITERATION = 35 * 1048576


def _job_stdouts(run_dir: str):
    """``[(mtime, path)]`` for ``slurm_stdout_<id>.log``, newest last."""
    out = []
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            if re.match(r"^slurm_stdout_\d+\.log$", fn):
                p = os.path.join(root, fn)
                try:
                    out.append((os.path.getmtime(p), p))
                except OSError:
                    continue
    out.sort()
    return out


def _short_members(run_dir: str, store: str, young_bytes: int):
    """The SHORT tar's member list: an ALLOWLIST, not a filtered denylist.

    User ruling 2026-09-28, after the audit measured the previous short
    tar at 129 MB of which ~95% was filtered dead-job and duplicate
    logs. What a reader of a short tar actually needs:

    * the NEWEST ``slurm_stdout_<id>.log`` RAW -- it is a verbatim
      SUPERSET of globalfit_run.log plus the rank logs, and it alone
      carries [GF_TIMING], [V9-SEED], [r4/saver], the stage table and
      the INFO-level [GB_CELL_LL] cold breakdown (only the WARNING
      variant matches LOG_KEEP_PATTERN, so the filtered path would
      lose it). It gzips ~11x, so raw is both cheaper to produce and
      more useful than a filtered copy.
    * the previous job's stdout TOO when the newest is younger than one
      saved iteration, so the tar always spans at least one.
    * the reduced extract (keep=1), the page if present, the newest
      job's gpu_util csv, run_settings.log, the *_setup.log files, the
      eigen pkl, the run's residual snapshot npz, and
      gb_fstat_fit/*/DONE.json.

    DROPPED: every globalfit_run* product (duplicated by the stdout),
    every dead-job stdout, gpu_procs (per-process noise), and the truth
    and warmstart npz (static, incompressible, identical across tars --
    94 MB on 6mo).

    NO _filtered/_tail products are written or shipped in short mode.
    """
    keep = []
    stdouts = _job_stdouts(run_dir)
    if stdouts:
        keep.append(stdouts[-1][1])
        if (len(stdouts) > 1
                and os.path.getsize(stdouts[-1][1]) < int(young_bytes)):
            keep.append(stdouts[-2][1])
            logger.info(
                "short snapshot: newest job is < one saved iteration "
                "(%.0f MB); shipping the previous job's stdout too",
                os.path.getsize(stdouts[-1][1]) / 1048576.0)
    _newest_job = None
    if stdouts:
        m = re.search(r"slurm_stdout_(\d+)\.log$", stdouts[-1][1])
        _newest_job = m.group(1) if m else None
    if store:
        ex = store[:-3] + "_extract.h5"
        if os.path.exists(ex):
            keep.append(ex)
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            p = os.path.join(root, fn)
            rel = "/" + os.path.relpath(p, run_dir)
            if fn == "gf_monitor.html":
                keep.append(p)
            elif fn == "run_settings.log" or fn.endswith("_setup.log"):
                keep.append(p)
            elif fn.endswith("_eigen_tables.pkl"):
                keep.append(p)
            elif fn.endswith("_residual_snapshot.npz"):
                # the run's own residual for the data/template/residual
                # panels (GF_RESIDUAL_SNAPSHOT_EVERY; every cold walker
                # since 2026-10-11, ~80 MB on the 9mo grid at 4 walkers)
                keep.append(p)
            elif fn == "DONE.json" and "gb_fstat_fit" in rel:
                keep.append(p)
            elif (_newest_job is not None
                  and fn == f"gpu_util_{_newest_job}.csv"):
                keep.append(p)
    # stable, de-duplicated
    seen, out = set(), []
    for p in keep:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(p)
    return out


#: Seconds an output-data snapshot waits for a residual snapshot that matches
#: the store before it ships the one it has (and warns).
_OUTPUT_DATA_WAIT_S = 1800.0


def _stored_log_like(store: str):
    """``(iteration, log_like)``: the store's stored row count and the cold
    chain's log likelihood of each stored row, walkers flattened."""
    import h5py

    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        it = int(g.attrs["iteration"])
        ll = g["log_like"][:it]
    return it, ll.reshape(it, -1)


def _output_data_rows(store: str, sidecar: Optional[str] = None,
                      frozen: Optional[str] = None):
    """``(iteration, keep, cold_keep)`` of an output-data extract, with the
    residual snapshot it ships copied to ``frozen``.

    The snapshot names its row (``store_row``) and carries that row's cold
    log likelihoods (``lnl``). It is shipped only once its ``lnl`` equals the
    stored row it names: the 9-month tars of 2026-10-22 paired a 2016-row
    extract with a snapshot of row 2021, twice, so the label alone is not
    trusted. Until they match it re-reads the store every 10 s for up to
    :data:`_OUTPUT_DATA_WAIT_S` and prints why every minute, with the age of
    each file's last write (a stopped run writes neither): a row not stored
    yet keeps the copy (the saver can lag the head, and the head writes newer
    snapshots meanwhile); a stored row with other log likelihoods (a snapshot
    left from an earlier trajectory) takes the run's next snapshot. Then it
    ships the copy it has and warns. ``iteration`` pins the rows the extract
    covers to the store at the match, so rows saved during the extract stay
    out. ``cold_keep`` is every row (the cold chains are the posterior
    samples); ``keep``, for the large tempered arrays, runs from the
    snapshot's row to the last row (1 without a snapshot).
    """
    import shutil

    import numpy as np

    if sidecar is None:
        it = _stored_log_like(store)[0]
        return it, 1, it

    def _age(p):
        try:
            return f"{(time.time() - os.path.getmtime(p)) / 60.0:.0f} min ago"
        except OSError:
            return "never"

    t0, said, copied, row, stale = time.monotonic(), None, None, None, True
    while True:
        it = None
        try:
            m = os.path.getmtime(sidecar)
            if stale and m != copied:
                shutil.copyfile(sidecar, frozen)
                copied = m
                with np.load(frozen) as z:
                    row, lnl = int(z["store_row"]), np.ravel(z["lnl"])
            it, ll = _stored_log_like(store)
            if row < it and np.array_equal(ll[row], lnl):
                return it, it - row, it
            stale = row < it
            why = (f"its row {row} is not stored yet" if row >= it
                   else f"its lnl does not match the stored row {row}")
            if ll.shape[1] == lnl.size:
                hits = np.flatnonzero(np.all(ll == lnl, axis=1))
                if hits.size:
                    why += f" (it matches the stored row {int(hits[0])})"
            why += f"; the store holds {it} rows"
        except (OSError, KeyError, ValueError) as e:
            copied, stale = None, True
            why = f"unreadable ({type(e).__name__}: {e})"
        now = time.monotonic()
        if now - t0 >= _OUTPUT_DATA_WAIT_S:
            logger.warning("output-data snapshot: the residual snapshot shipped does NOT "
                           "match the extract: %s", why)
            it = _stored_log_like(store)[0] if it is None else it
            return it, (it - row if row is not None and 0 <= row < it else 1), it
        if said is None or now - said >= 60.0:
            said = now
            print(f"[monitor] output-data: waiting for a residual snapshot that matches "
                  f"the store ({(now - t0) / 60:.0f} of {_OUTPUT_DATA_WAIT_S / 60:.0f} "
                  f"min): {why}; the store was last written {_age(store)}, the "
                  f"snapshot {_age(sidecar)}", flush=True)
        time.sleep(10.0)


def _members(run_dir: str, include_fstat: bool, skip_raw_log,
             max_file_mb: Optional[int] = None, keep_products=()):
    """Archive members. ``skip_raw_log`` is one path or a set of them.

    ``max_file_mb`` drops anything bigger (short mode). The reduced
    extract ``.h5`` is EXEMPT: it is "the most recent state", i.e. the
    point of the snapshot, and it is already bounded by ``keep`` /
    ``cold_keep`` rather than by a byte count.
    """
    if skip_raw_log is None:
        skip = set()
    elif isinstance(skip_raw_log, (set, frozenset, list, tuple)):
        skip = {os.path.abspath(p) for p in skip_raw_log}
    else:
        skip = {os.path.abspath(skip_raw_log)}
    keep_products = {os.path.abspath(p) for p in (keep_products or ())}
    out = []
    dropped = []
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            p = os.path.join(root, fn)
            rel = "/" + os.path.relpath(p, run_dir)
            if any(part in rel for part in _EXCLUDE_PATH_PARTS):
                continue
            if any(fn.endswith(s) for s in _EXCLUDE_SUFFIXES):
                continue
            if ".h5.bak" in fn:
                continue
            # STALE REDUCED LOGS. The full build deliberately ships the
            # globalfit_run_filtered/_tail PAIR when the run log is over
            # the cap -- _filter_run_log writes it and hands its raw
            # path back as ``skip``, and SnapshotMatchesTheShellRecipeTest
            # pins that. But the OLD short mode left
            # slurm_stdout_<id>_filtered/_tail lying in the run dir
            # root, and the full build then shipped them BESIDE the raw
            # logs -- pure duplication. So: keep the products THIS call
            # produced (``keep_products``), drop every other one.
            if (fn.endswith(("_filtered.log", "_tail.log"))
                    and os.path.abspath(p) not in keep_products):
                continue
            # Full stores out, the reduced extract in.
            if fn.endswith(".h5") and not fn.endswith("_extract.h5"):
                continue
            if not include_fstat and "gb_fstat_fit" in rel and fn != "DONE.json":
                continue
            if os.path.abspath(p) in skip:
                continue
            # EXEMPT from the size cap: the reduced products we just went
            # to the trouble of making. The extract IS "the most recent
            # state" and is bounded by keep/cold_keep; a _filtered.log is
            # bounded by the keep-pattern and a _tail.log by tail_mb.
            # Without this the cap silently discards the filtered log of
            # a big job and leaves only its tail -- caught on a real 6mo
            # run dir, where slurm_stdout_654_filtered.log was built and
            # then dropped while its tail survived.
            if max_file_mb is not None and not fn.endswith(
                    ("_extract.h5", "_filtered.log", "_tail.log")):
                try:
                    mb = os.path.getsize(p) / 1048576.0
                except OSError:
                    continue
                if mb > max_file_mb:
                    dropped.append((rel, mb))
                    continue
            out.append(p)
    if dropped:
        dropped.sort(key=lambda x: -x[1])
        logger.info(
            "short snapshot: dropped %d file(s) over %d MB (%s%s)",
            len(dropped), max_file_mb,
            ", ".join(f"{r} {m:.0f} MB" for r, m in dropped[:4]),
            ", ..." if len(dropped) > 4 else "")
    return out


def build_snapshot(run_dir: str, out_path: Optional[str] = None, *,
                   keep: Optional[int] = None,
                   cold_keep: Optional[int] = None,
                   include_fstat: bool = False,
                   log_cap_mb: int = 200,
                   short: bool = False,
                   young_bytes: Optional[int] = None,
                   max_file_mb: Optional[int] = None,
                   tail_mb: int = 5,
                   output_data: bool = False) -> Optional[str]:
    """Build ``<run_dir>_snapshot.tar.gz``. Returns the path, or ``None``.

    ``short=True`` builds ``<run_dir>_short.tar.gz`` instead: log
    information and the most recent state, and nothing big. Concretely
    it overrides the defaults to ``keep=1``, ``cold_keep=1``,
    ``log_cap_mb=20`` and ``max_file_mb=5``, and reduces EVERY oversized
    ``*.log`` to its filtered + tail pair rather than only the run log.
    ``include_fstat`` stays off, as it already is by default. Any of
    those can still be passed explicitly to override the preset.

    What that removes, measured on 6mo v9: the per-job
    ``slurm_stdout_*.log`` (386 MB for job 650, shipped whole before
    because only ``*_artifacts/globalfit_run.log`` was ever filtered),
    ``gb_truth_3to21.npz`` (78 MB), and the five-iteration extract in
    favour of a one-iteration one. A full snapshot of that run is
    ~330 MB compressed.

    ``output_data=True`` builds ``<run_dir>_output_data.tar.gz``: the store
    extract, the residual snapshot and the run folder's WDM lookup tables
    only (see the module docstring). It ships a copy of the residual
    snapshot that matches the store and pins the extract's rows to what the
    store holds then (see :func:`_output_data_rows`): ``cold_keep``
    defaults to every stored row and ``keep`` to the rows from the
    snapshot's ``store_row`` to the store's last row; without a snapshot
    file it ships the extract alone with a one-row ``keep``, and a warning. It
    cannot be combined with ``short`` (returns ``None`` with a warning).

    Never raises: this runs on the writer rank, where a failed diagnostic
    must not become a failed run. A ``None`` return with a warning is the
    contract.
    """
    run_dir = os.path.abspath(str(run_dir).rstrip("/"))
    if short and output_data:
        logger.warning("snapshot NOT built: short and output_data are two "
                       "different tars; ask for one of them.")
        return None
    # ⚠ REAL SENTINELS, not value comparison. These used to default to
    # 5 / 12 and short mode overrode them with ``if keep == 5``, which
    # makes an EXPLICIT keep=5 indistinguishable from the default and
    # silently demotes it to 1 -- caught by the test that asserts an
    # explicit kwarg still wins. ``None`` means "caller said nothing".
    # output_data resolves its own defaults from the store, below.
    if not output_data:
        if keep is None:
            keep = 1 if short else 5
        if cold_keep is None:
            cold_keep = 1 if short else 12
    out_path = out_path or (run_dir + (
        "_short.tar.gz" if short
        else "_output_data.tar.gz" if output_data
        else "_snapshot.tar.gz"))
    st = time.perf_counter()
    # output_data: the frozen copy of the residual snapshot, its name in the
    # tar, and the stored row count the extract is pinned to
    frozen, arcnames, _it = None, {}, None
    try:
        store = _live_store(run_dir)
        if store is None:
            logger.warning("snapshot: no live *testing*.h5 under %s", run_dir)
            return None
        if output_data:
            from ..residual_snapshot import snapshot_path

            sidecar = snapshot_path(store)
            if os.path.exists(sidecar):
                # The run overwrites the snapshot every few rows: ship a copy
                # that matches the store, the one the extract is pinned to.
                frozen = out_path + ".residual_snapshot.tmp"
                arcnames[frozen] = os.path.relpath(sidecar, os.path.dirname(run_dir))
            else:
                logger.warning("output-data snapshot: no residual snapshot "
                               "%s; shipping the store extract alone",
                               sidecar)
            _it, _keep, _cold_keep = _output_data_rows(
                store, sidecar if frozen else None, frozen)
            if keep is None:
                keep = _keep
            if cold_keep is None:
                cold_keep = _cold_keep
        from ._store_extract import extract

        extract(store, store[:-3] + "_extract.h5", keep, cold_keep=cold_keep,
                **(dict(gzip_level=9, shuffle=True, iteration=_it)
                   if output_data else {}))
        if output_data:
            # the run's WDM lookup table(s): the EMRI direct template and the SOBBH
            # lookup likelihood read them, so regenerating those sources needs them
            tables = sorted(os.path.join(run_dir, fn) for fn in os.listdir(run_dir)
                            if fn.startswith("wdm_lookup_") and fn.endswith(".h5"))
            print("[monitor] output-data: writing the tar", flush=True)
            members = ([store[:-3] + "_extract.h5"] + ([frozen] if frozen else [])
                       + tables)
        elif short:
            # ALLOWLIST. No log filtering at all: the newest stdout
            # ships RAW (it gzips ~11x and is a superset of every other
            # log), and nothing writes _filtered/_tail products into
            # the run dir -- the old short mode did, and the next FULL
            # build then shipped them beside the raw logs.
            members = _short_members(
                run_dir, store,
                _STDOUT_BYTES_PER_ITERATION if young_bytes is None
                else int(young_bytes))
        else:
            skip = _filter_run_log(run_dir, log_cap_mb)
            # The pair THIS call produced, if it produced one; every
            # other *_filtered/_tail in the tree is stale and dropped.
            _kp = ()
            if skip:
                _b = skip[:-4] if skip.endswith(".log") else skip
                _kp = (_b + "_filtered.log", _b + "_tail.log")
            members = _members(run_dir, include_fstat, skip,
                               max_file_mb=max_file_mb, keep_products=_kp)
        # Write to a temp and rename: a consumer polling for the tar must
        # never pick up a partial archive. Same reasoning as the page and
        # the running backup copy.
        tmp = out_path + ".tmp"
        with tarfile.open(tmp, "w:gz") as tf:
            for p in members:
                try:
                    tf.add(p, arcname=arcnames.get(p) or os.path.relpath(
                        p, os.path.dirname(run_dir)))
                except (FileNotFoundError, OSError) as e:
                    # A live file can vanish or change mid-archive. Benign:
                    # the append-only log gives a consistent prefix and the
                    # extract h5 is written atomically.
                    logger.debug("snapshot: skipping %s (%s)", p, e)
        os.replace(tmp, out_path)
    except Exception as e:                       # noqa: BLE001 -- see docstring
        logger.warning("snapshot NOT built (%s: %s). The run is unaffected.",
                       type(e).__name__, e)
        for _p in (out_path + ".tmp", frozen):
            try:
                if _p and os.path.exists(_p):
                    os.remove(_p)
            except Exception:
                pass
        return None
    _mb = os.path.getsize(out_path) / 1048576.0
    if short or output_data:
        # THE MEMBER LIST, not just the size. Both are allowlists, so
        # "what did it decide to ship" is the thing that goes wrong, and
        # it is invisible from the size alone.
        _rows = []
        for _p in members:
            try:
                _rows.append((os.path.getsize(_p) / 1048576.0, arcnames.get(_p)
                              or os.path.relpath(_p, os.path.dirname(run_dir))))
            except OSError:
                continue
        _rows.sort(reverse=True)
        logger.info(
            "%s snapshot %s: %.1f MB compressed from %d member(s), "
            "%.1f MB raw --\n%s", "short" if short else "output-data",
            out_path, _mb, len(_rows),
            sum(r[0] for r in _rows),
            "\n".join(f"    {r[0]:8.2f} MB  {r[1]}" for r in _rows))
    logger.info("snapshot %s (%.1f MB) in %.1f s", out_path, _mb,
                time.perf_counter() - st)
    try:
        if frozen and os.path.exists(frozen):
            os.remove(frozen)
    except OSError:
        pass
    return out_path
