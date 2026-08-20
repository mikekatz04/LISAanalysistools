"""Host-memory watchdog for cupy/mpi4py jobs killed by the Slurm cgroup OOM killer.

The cgroup OOM killer sends SIGKILL, which Python cannot catch: no exception,
no traceback, no atexit. The only way to get diagnostics is to observe memory
growth *before* the limit is reached.

Usage in run_global.py, immediately after parsing args:

    from host_mem_watchdog import setup_logging, start_watchdog, log_mem

    setup_logging()
    start_watchdog(warn_frac=0.75, dump_frac=0.85, abort_frac=0.92)

and, if you have access to the iteration loop, call ``log_mem(i)`` once per
iteration. If you do not, the background thread alone is enough -- it samples
on its own timer.
"""

from __future__ import annotations

import gc
import linecache
import logging
import os
import signal
import sys
import threading
import time
import tracemalloc

logger = logging.getLogger("memwatch")

_GIB = 1024 ** 3


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO, stream=sys.stderr, rank=None) -> None:
    """Configure the root logger so records are actually emitted and flushed.

    Without this, ``logger.debug(...)`` calls are silently discarded because the
    root logger defaults to WARNING with no handler attached.

    Pass ``rank`` under MPI so every line is attributable to a process.
    """
    handler = logging.StreamHandler(stream)
    tag = f" rank{rank}" if rank is not None else ""
    handler.setFormatter(
        logging.Formatter(f"[%(asctime)s{tag}] %(levelname)s %(name)s: %(message)s")
    )

    logging.basicConfig(level=level, handlers=[handler], force=True)

    # Make every emit flush, so nothing is lost when the process is SIGKILLed.
    _orig_emit = handler.emit

    def _emit(record):
        _orig_emit(record)
        handler.flush()

    handler.emit = _emit


# --------------------------------------------------------------------------
# memory readings
# --------------------------------------------------------------------------
def _read_int(path: str):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _cgroup_paths():
    """Return (current_path, max_path) for cgroup v2 or v1, or (None, None)."""
    # cgroup v2, namespaced (common inside Slurm job containers)
    if os.path.exists("/sys/fs/cgroup/memory.current"):
        return "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"

    # cgroup v2, non-namespaced: resolve our own path
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    base = os.path.join("/sys/fs/cgroup", parts[2].lstrip("/"))
                    cur = os.path.join(base, "memory.current")
                    if os.path.exists(cur):
                        return cur, os.path.join(base, "memory.max")
    except OSError:
        pass

    # cgroup v1
    v1 = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    if os.path.exists(v1):
        return v1, "/sys/fs/cgroup/memory/memory.limit_in_bytes"

    return None, None


_CG_CUR, _CG_MAX = _cgroup_paths()


def cgroup_usage_bytes():
    return _read_int(_CG_CUR) if _CG_CUR else None


def cgroup_limit_bytes():
    if not _CG_MAX:
        return None
    try:
        with open(_CG_MAX) as f:
            raw = f.read().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    # cgroup v1 uses a huge sentinel for "unlimited"
    return None if val > (1 << 62) else val


def rss_bytes():
    """Resident set size of this process, from /proc/self/statm."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None


def cupy_pool_stats():
    """Device and pinned-host pool usage. Returns a dict, possibly empty."""
    try:
        import cupy as cp
    except Exception:
        return {}

    out = {}
    try:
        dev = cp.get_default_memory_pool()
        out["dev_used_gib"] = dev.used_bytes() / _GIB
        out["dev_total_gib"] = dev.total_bytes() / _GIB
    except Exception:
        pass
    try:
        pinned = cp.get_default_pinned_memory_pool()
        # n_free_blocks() is the only public counter; growth here is a strong
        # signal that pinned HOST memory is the leak.
        out["pinned_free_blocks"] = pinned.n_free_blocks()
    except Exception:
        pass
    return out


_CG_STAT = (
    os.path.join(os.path.dirname(_CG_CUR), "memory.stat") if _CG_CUR else None
)

# anon      -> ordinary process allocations (numpy, Python objects)
# file      -> page cache from job I/O; usually reclaimable before an OOM kill
# unevictable / mlock -> PINNED host memory. cupy's pinned pool lands here.
#              Steady growth in this counter is the smoking gun for a pinned
#              memory leak driven by device<->host transfers.
_CG_STAT_KEYS = ("anon", "file", "slab", "unevictable", "mlock",
                 "total_rss", "total_cache", "total_unevictable")


def cgroup_stat_breakdown():
    """Parse memory.stat into a dict of the interesting counters, in bytes."""
    if not _CG_STAT:
        return {}
    try:
        with open(_CG_STAT) as f:
            raw = dict(
                (parts[0], int(parts[1]))
                for parts in (line.split() for line in f)
                if len(parts) == 2 and parts[1].isdigit()
            )
    except (OSError, ValueError):
        return {}
    return {k: raw[k] for k in _CG_STAT_KEYS if k in raw}


def free_pinned_pool():
    """Release cached pinned host memory. Safe to call between iterations."""
    try:
        import cupy as cp

        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
COUNT_GC_OBJECTS = False  # gc.get_objects() walks the entire heap; see docs below


def mem_summary(tag: str = "") -> str:
    rss = rss_bytes()
    cur = cgroup_usage_bytes()
    lim = cgroup_limit_bytes()
    parts = [f"[mem{' ' + tag if tag else ''}]"]
    if rss is not None:
        parts.append(f"rss={rss / _GIB:.2f}G")
    if cur is not None:
        frac = f" ({100 * cur / lim:.0f}% of {lim / _GIB:.0f}G)" if lim else ""
        parts.append(f"cgroup={cur / _GIB:.2f}G{frac}")
    for k, v in cgroup_stat_breakdown().items():
        parts.append(f"{k}={v / _GIB:.2f}G")
    for k, v in cupy_pool_stats().items():
        parts.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
    if COUNT_GC_OBJECTS:
        parts.append(f"gc_objects={len(gc.get_objects())}")
    return "  ".join(parts)


def log_mem(tag="") -> None:
    logger.info(mem_summary(str(tag)))


_tm_baseline = None


def tracemalloc_report(top: int = 15) -> None:
    """Log the largest Python-level allocation sites, and the delta since the
    previous report. Only sees Python allocations -- not cupy pinned memory or
    C-extension buffers -- but it catches accumulating lists/dicts/arrays."""
    if not tracemalloc.is_tracing():
        logger.info("tracemalloc not enabled; start_watchdog(trace=True) to enable")
        return

    global _tm_baseline
    snap = tracemalloc.take_snapshot().filter_traces(
        (tracemalloc.Filter(False, tracemalloc.__file__),)
    )

    logger.info("--- tracemalloc: top %d allocation sites ---", top)
    for stat in snap.statistics("lineno")[:top]:
        frame = stat.traceback[0]
        line = linecache.getline(frame.filename, frame.lineno).strip()
        logger.info(
            "  %8.1f MB  %5d blocks  %s:%d  %s",
            stat.size / 1024 ** 2, stat.count, frame.filename, frame.lineno, line,
        )

    if _tm_baseline is not None:
        logger.info("--- tracemalloc: growth since baseline ---")
        for stat in snap.compare_to(_tm_baseline, "lineno")[:top]:
            if stat.size_diff <= 0:
                continue
            frame = stat.traceback[0]
            logger.info(
                "  +%8.1f MB  %+6d blocks  %s:%d",
                stat.size_diff / 1024 ** 2, stat.count_diff,
                frame.filename, frame.lineno,
            )
    _tm_baseline = snap


def find_names_for_array(arr, max_depth: int = 3):
    """Best-effort: find variable names referencing this array.

    Walks gc referrers looking for dicts that are a frame's locals or an
    instance ``__dict__``. Slow and incomplete -- use only in a dump path,
    never on a per-iteration timer.
    """
    names = []
    seen = set()

    def search(obj, depth):
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        for ref in gc.get_referrers(obj):
            if isinstance(ref, dict):
                for k, v in ref.items():
                    if v is obj:
                        for ref2 in gc.get_referrers(ref):
                            if getattr(ref2, "f_locals", None) is ref:
                                names.append(
                                    f"local {k!r} in {ref2.f_code.co_name}() "
                                    f"line {ref2.f_lineno}"
                                )
                            elif getattr(ref2, "__dict__", None) is ref:
                                names.append(
                                    f"attribute {k!r} of {type(ref2).__name__}"
                                )
            elif isinstance(ref, (list, tuple)):
                search(ref, depth + 1)

    search(arr, 0)
    return names


def dump_gpu_arrays(min_size_mb: float = 5.0, names: bool = False) -> None:
    """Log live cupy arrays above a size threshold.

    ``names=True`` additionally tries to resolve each array back to a variable
    name. That is expensive, so reserve it for the GPU-OOM path.
    """
    try:
        import cupy as cp
    except Exception:
        return
    rows = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, cp.ndarray) and obj.nbytes / 1024 ** 2 >= min_size_mb:
                rows.append(
                    (obj.nbytes / 1024 ** 2, obj.shape, obj.dtype,
                     find_names_for_array(obj) if names else None)
                )
        except ReferenceError:
            continue
    rows.sort(reverse=True, key=lambda r: r[0])
    logger.warning("--- live cupy arrays >= %.0f MB (%d) ---", min_size_mb, len(rows))
    for size_mb, shape, dtype, refs in rows[:40]:
        suffix = f"  -> {refs}" if names else ""
        logger.warning("  %8.1f MB  %-25s %s%s", size_mb, str(shape), dtype, suffix)


def full_dump(reason: str) -> None:
    logger.warning("=== memory dump: %s ===", reason)
    logger.warning(mem_summary("dump"))
    tracemalloc_report()
    dump_gpu_arrays()
    logger.warning("=== end memory dump ===")


# --------------------------------------------------------------------------
# watchdog thread
# --------------------------------------------------------------------------
def start_watchdog(
    interval: float = 30.0,
    warn_frac: float = 0.75,
    dump_frac: float = 0.85,
    abort_frac: float | None = 0.92,
    trace: bool = True,
    limit_bytes: int | None = None,
) -> threading.Thread:
    """Start a daemon thread that samples memory and reports before the kill.

    warn_frac  -- log a warning line at this fraction of the limit
    dump_frac  -- emit a full diagnostic dump (once) at this fraction
    abort_frac -- send SIGINT to ourselves at this fraction, turning an
                  uncatchable SIGKILL into a KeyboardInterrupt with a real
                  Python traceback showing where the job was. Set to None to
                  disable.
    trace      -- enable tracemalloc (adds roughly 10-30% overhead to Python
                  allocations; negligible for GPU-bound work)
    """
    if trace and not tracemalloc.is_tracing():
        tracemalloc.start(10)

    limit = limit_bytes or cgroup_limit_bytes()
    if limit is None:
        logger.warning(
            "watchdog: could not read cgroup limit; pass limit_bytes explicitly "
            "(e.g. the value of --mem). Falling back to RSS-only logging."
        )

    state = {"warned": False, "dumped": False, "aborted": False}

    def _loop():
        logger.info("watchdog started: %s", mem_summary("start"))
        tracemalloc_report(top=5)  # establish the baseline
        while True:
            time.sleep(interval)
            cur = cgroup_usage_bytes() or rss_bytes()
            if cur is None:
                continue
            logger.info(mem_summary("tick"))
            if limit is None:
                continue
            frac = cur / limit
            if abort_frac and frac >= abort_frac and not state["aborted"]:
                state["aborted"] = True
                full_dump(f"{100 * frac:.0f}% of cgroup limit -- aborting")
                logger.error("watchdog: sending SIGINT to obtain a traceback")
                os.kill(os.getpid(), signal.SIGINT)
            elif frac >= dump_frac and not state["dumped"]:
                state["dumped"] = True
                full_dump(f"{100 * frac:.0f}% of cgroup limit")
            elif frac >= warn_frac and not state["warned"]:
                state["warned"] = True
                logger.warning("watchdog: at %.0f%% of cgroup limit", 100 * frac)

    t = threading.Thread(target=_loop, name="memwatch", daemon=True)
    t.start()
    return t