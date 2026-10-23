"""Per-process and per-node host-memory telemetry for a running global fit.

Why (2026-10-20): the 9mo production jobs 795 and 798 were both SIGKILLed by
the kernel's out-of-memory killer on node B after ~8 h (iteration ~70, inside
the EMRI update), with no traceback and no memory line in any log. The
compute ranks' own peak RSS (the ``[host maxRSS ...]`` of the SubBandBuffer
lines) was flat at 71-75 GB for the last 6 h, the monitor rank was idle, and
the trajectory-pool workers do not grow per call, so the logs cannot say
whether one process blew up in its last minute or the node's memory went
somewhere no process owns (tmpfs, unreclaimable slab). This module is the
flight recorder for that.

Every MPI process of the run (head, compute ranks, saver, monitor rank)
starts one daemon thread (:func:`start`) that reads ``/proc`` every
``GF_MEMWATCH_PERIOD`` seconds (default 30; ``GF_MEMWATCH_FAST_PERIOD``, 5,
while the node's available memory is below ``GF_MEMWATCH_PRESS_FRAC``, 0.30,
of its total) and logs

``[GF_MEM] w<rank> pid <pid> on <host>: rss X.X GB, children Y.Y GB (n),
node avail Z.Z of T.T GB``

whenever its own RSS or its children's (the EMRI trajectory-pool workers)
moved by more than ``GF_MEMWATCH_DELTA_GB`` (2) since its last line, the
node's available memory moved by more than twice that, or
``GF_MEMWATCH_HEARTBEAT_S`` (600) elapsed. One process per node (local rank
0, read from ``MPI_LOCALRANKID`` / ``PMI_LOCAL_RANK`` /
``OMPI_COMM_WORLD_LOCAL_RANK`` / ``MV2_COMM_WORLD_LOCAL_RANK`` /
``SLURM_LOCALID``; every process when none is set) logs the node's largest
processes of the run's user, with the node's anonymous / shmem /
page-cache / unreclaimable-slab / dirty totals, every ``GF_MEMWATCH_TOP_S``
(600).

Below ``GF_MEMWATCH_LOW_FRAC`` (0.15) of the node available ("LOW MEMORY"):
every process logs its own line and the node table is logged every
``GF_MEMWATCH_LOW_S`` (15) seconds by whichever process gets there first,
and the largest REGISTERED process of the node (one that called
:func:`start`; the registry is a per-job directory of pid files) plus the
registered parent of the largest process overall are sent ``SIGWINCH`` at
most once per ``GF_MEMWATCH_DUMP_GAP_S`` (120) each: :func:`start` registers
:mod:`faulthandler` on that signal, so the process writes every thread's
Python stack to its stderr (the slurm log) from inside the signal handler,
even while it holds the GIL in a long C call. ``SIGWINCH`` is ignored by
default, so a process that never registered is not harmed by it.

The thread only reads ``/proc`` and touches small files in the registry
(no MPI, no GPU, no numpy) and never raises. Off Linux (no
``/proc/meminfo``) :func:`start` is a no-op. ``GF_MEMWATCH=0`` turns it off;
``GF_MEMWATCH_DUMP=0`` keeps the lines but sends no signal.

Where it runs: the production driver (``scripts/fstat_proposal/run_combined_staged.py``,
``main()``) calls ``start`` with the tag ``w`` + the MPI world rank on every process,
before the monitor-rank split, so the 9mo / 12mo runs have it on every
rank; neither launcher sets a ``GF_MEMWATCH*`` knob, so the defaults above
apply. The lines go through this module's :mod:`logging` logger
(``lisatools.globalfit.memwatch``, INFO). The registry
(:func:`default_registry_dir`, ``/tmp/gf_memwatch_{SLURM_JOB_ID}``) holds one
``pid_{pid}`` file per registered process (its tag) and the ``table_*`` /
``dump_*`` claim files of the LOW MEMORY path; nothing removes them.

Knobs (read once, by :func:`start`): ``GF_MEMWATCH`` (default 1),
``GF_MEMWATCH_DUMP`` (1), ``GF_MEMWATCH_PERIOD`` (30 s),
``GF_MEMWATCH_FAST_PERIOD`` (5 s), ``GF_MEMWATCH_PRESS_FRAC`` (0.30),
``GF_MEMWATCH_DELTA_GB`` (2), ``GF_MEMWATCH_HEARTBEAT_S`` (600),
``GF_MEMWATCH_TOP_S`` (600), ``GF_MEMWATCH_LOW_FRAC`` (0.15),
``GF_MEMWATCH_LOW_S`` (15), ``GF_MEMWATCH_DUMP_GAP_S`` (120). An unreadable
numeric value falls back to its default. Sizes are GiB (``2**30`` bytes,
:data:`GB`).
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["MemWatch", "start", "read_meminfo", "read_meminfo_fields", "proc_table", "GB"]

GB = 1024.0 ** 3
_LOCAL_RANK_ENVS = ("MPI_LOCALRANKID", "PMI_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
                    "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID")
# the node totals of the LOW MEMORY / periodic table: where memory sits that no
# process's RSS shows (Shmem = tmpfs + shared anonymous, SUnreclaim = kernel)
_NODE_FIELDS = (("AnonPages", "anon"), ("Shmem", "shmem"), ("Cached", "page cache"),
                ("SUnreclaim", "slab unreclaimable"), ("Dirty", "dirty"))
_DUMP_SIGNAL = getattr(signal, "SIGWINCH", None)
_STARTED: Optional["MemWatch"] = None


def _env_float(name: str, default: float) -> float:
    """``float(os.environ[name])``, or ``default`` when unset or unreadable."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def read_meminfo_fields(root: str = "/proc") -> Optional[Dict[str, float]]:
    """Every ``/proc/meminfo`` field in bytes, or ``None`` when unreadable.

    A value with a unit (``kB``) is multiplied by 1024; a bare count (the
    ``HugePages_*`` lines) is kept as is.

    Args:
        root: The ``/proc`` mount (tests point it at a fake tree).

    Returns:
        dict[str, float] or None: ``{field: value}``; ``None`` when the file
        cannot be read or lacks ``MemTotal`` / ``MemAvailable``.
    """
    vals: Dict[str, float] = {}
    try:
        with open(os.path.join(root, "meminfo")) as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        vals[key.strip()] = float(parts[0]) * (1024.0 if len(parts) > 1 else 1.0)
                    except ValueError:
                        continue
    except OSError:
        return None
    if "MemTotal" not in vals or "MemAvailable" not in vals:
        return None
    return vals


def read_meminfo(root: str = "/proc") -> Optional[Tuple[float, float]]:
    """``(MemTotal, MemAvailable)`` in bytes, or ``None`` when unreadable.

    Args:
        root: The ``/proc`` mount.

    Returns:
        tuple[float, float] or None: Total and available memory, bytes.
    """
    vals = read_meminfo_fields(root)
    if vals is None:
        return None
    return vals["MemTotal"], vals["MemAvailable"]


def _read_status(root: str, pid: str) -> Optional[Tuple[float, int, int]]:
    """``(VmRSS bytes, real uid, ppid)`` of ``pid``, or ``None``."""
    rss, uid, ppid = 0.0, -1, -1
    try:
        with open(os.path.join(root, pid, "status")) as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) * 1024.0
                elif line.startswith("Uid:"):
                    uid = int(line.split()[1])
                elif line.startswith("PPid:"):
                    ppid = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return rss, uid, ppid


def _cmdline(root: str, pid: str, width: int = 90) -> str:
    """``pid``'s command line for the node table, cut to ``width`` characters.

    A leading ``python*`` interpreter path is dropped; ``"?"`` when unreadable or empty.
    """
    try:
        with open(os.path.join(root, pid, "cmdline"), "rb") as fh:
            raw = fh.read(4096)
    except OSError:
        return "?"
    parts = [p for p in raw.decode("utf-8", "replace").split("\x00") if p]
    # drop the interpreter path, keep the script / module and its first args
    if len(parts) > 1 and os.path.basename(parts[0]).startswith("python"):
        parts = parts[1:]
    return " ".join(parts)[:width] or "?"


def proc_table(root: str = "/proc", uid: Optional[int] = None) -> List[Tuple[int, int, float]]:
    """``[(pid, ppid, rss_bytes), ...]`` of every readable process (of ``uid`` if given).

    Args:
        root: The ``/proc`` mount.
        uid: Keep only processes of this real uid. Default ``None``: all.

    Returns:
        list[tuple[int, int, float]]: One row per process whose ``status``
        could be read (empty when ``root`` cannot be listed).
    """
    out = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        st = _read_status(root, name)
        if st is None:
            continue
        rss, puid, ppid = st
        if uid is not None and puid != uid:
            continue
        out.append((int(name), ppid, rss))
    return out


def _descendants_rss(table, pid: int) -> Tuple[float, int]:
    """Total RSS and count of every descendant of ``pid`` in ``table``."""
    kids: Dict[int, List[Tuple[int, float]]] = {}
    for p, pp, rss in table:
        kids.setdefault(pp, []).append((p, rss))
    total, n, stack, seen = 0.0, 0, [pid], {pid}
    while stack:
        for c, rss in kids.get(stack.pop(), ()):
            if c in seen:
                continue
            seen.add(c)
            total += rss
            n += 1
            stack.append(c)
    return total, n


def _local_rank() -> Optional[int]:
    """The node-local rank from the first set integer of ``_LOCAL_RANK_ENVS``, else ``None``."""
    for k in _LOCAL_RANK_ENVS:
        v = os.environ.get(k)
        if v is not None and str(v).strip().lstrip("-").isdigit():
            return int(v)
    return None


def default_registry_dir() -> str:
    """The per-job pid registry / lock directory (node-local ``/tmp``).

    Returns:
        str: ``/tmp/gf_memwatch_{SLURM_JOB_ID}`` (``local`` when the job id
        is unset).
    """
    job = os.environ.get("SLURM_JOB_ID") or "local"
    return os.path.join("/tmp", f"gf_memwatch_{job}")


class MemWatch(threading.Thread):
    """The sampling thread (see the module docstring). ``sample_once`` is the testable step.

    A daemon :class:`threading.Thread` named ``gf-memwatch``. :func:`start`
    builds one per process from the ``GF_MEMWATCH_*`` knobs; tests build it
    directly on a fake ``/proc`` tree with fake clocks and a fake ``kill``.

    Args:
        tag: The process label in every line (``w5`` = world rank 5).
        root: The ``/proc`` mount. Default ``"/proc"``.
        period: Seconds between samples. Default 30.
        fast_period: Seconds between samples under pressure. Default 5.
        press_frac: Available / total below which the fast period applies.
            Default 0.30.
        delta_gb: Change in own / children RSS (GiB) that triggers a line;
            twice it for the node's available memory. Default 2.
        heartbeat_s: Seconds after which a line is logged anyway. Default 600.
        top_s: Seconds between the node-reporter's tables. Default 600.
        low_frac: Available / total below which LOW MEMORY applies.
            Default 0.15.
        low_s: LOW MEMORY cadence (own lines, node table bucket), seconds.
            Default 15.
        dump_gap_s: Wall-clock bucket of the stack-dump signal per target,
            seconds. Default 120.
        node_reporter: Whether this process logs the periodic node table.
            Default ``True``.
        top_n: Processes listed in a node table. Default 10.
        registry_dir: The pid registry / lock directory; ``None`` disables
            the registry (then the node reporter claims every table and no
            stack-dump signal has a target).
        dump_signal (signal.Signals or None): Signal sent for a stack dump.
            Default ``SIGWINCH`` (``None`` where the platform lacks it);
            ``None`` sends nothing.
        kill (callable): ``kill(pid, sig)``. Default :func:`os.kill`.
        clock (callable): Monotonic clock for the cadences. Default
            :func:`time.monotonic`.
        wall (callable): Wall clock for the cross-process buckets. Default
            :func:`time.time`.
        log (callable, optional): Line sink. Default the module logger's
            ``info``.

    Attributes:
        pid (int): This process's pid.
        uid (int or None): Its real uid (the node table's filter).
        host (str): The host name.
        pressure (bool): Whether the last sample was below ``press_frac``
            (selects the fast period).
    """

    def __init__(self, tag: str, *, root: str = "/proc", period: float = 30.0,
                 fast_period: float = 5.0, press_frac: float = 0.30,
                 delta_gb: float = 2.0, heartbeat_s: float = 600.0, top_s: float = 600.0,
                 low_frac: float = 0.15, low_s: float = 15.0, dump_gap_s: float = 120.0,
                 node_reporter: bool = True, top_n: int = 10, registry_dir: Optional[str] = None,
                 dump_signal=_DUMP_SIGNAL, kill=os.kill, clock=time.monotonic, wall=time.time,
                 log=None):
        super().__init__(name="gf-memwatch", daemon=True)
        self.tag = str(tag)
        self.root = root
        self.period = float(period)
        self.fast_period = float(fast_period)
        self.press_frac = float(press_frac)
        self.delta = float(delta_gb) * GB
        self.heartbeat_s = float(heartbeat_s)
        self.top_s = float(top_s)
        self.low_frac = float(low_frac)
        self.low_s = float(low_s)
        self.dump_gap_s = float(dump_gap_s)
        self.node_reporter = bool(node_reporter)
        self.top_n = int(top_n)
        self.registry_dir = registry_dir
        self.dump_signal = dump_signal
        self.kill = kill
        self.clock = clock
        self.wall = wall
        self.log = log or logger.info
        self.pid = os.getpid()
        self.uid = os.getuid() if hasattr(os, "getuid") else None
        self.host = socket.gethostname()
        self.pressure = False
        self._last: Optional[Tuple[float, float, float]] = None
        self._last_t: Optional[float] = None
        self._last_top_t: Optional[float] = None
        self._halt = threading.Event()

    # -- the per-job registry: who can take a stack-dump signal; cross-process locks --
    def register(self) -> None:
        """Add this process's pid to the registry (best effort)."""
        if not self.registry_dir:
            return
        try:
            os.makedirs(self.registry_dir, exist_ok=True)
            with open(os.path.join(self.registry_dir, f"pid_{self.pid}"), "w") as fh:
                fh.write(self.tag)
        except OSError:
            pass

    def _registered(self) -> Dict[int, str]:
        """``{pid: tag}`` of every process in the registry (empty without one)."""
        out: Dict[int, str] = {}
        if not self.registry_dir:
            return out
        try:
            names = os.listdir(self.registry_dir)
        except OSError:
            return out
        for name in names:
            if name.startswith("pid_") and name[4:].isdigit():
                try:
                    with open(os.path.join(self.registry_dir, name)) as fh:
                        out[int(name[4:])] = fh.read(64).strip()
                except OSError:
                    continue
        return out

    def _claim(self, name: str) -> bool:
        """``True`` for the first process of the node to claim ``name`` (an O_EXCL file).

        Without a registry, or when the file cannot be created for another
        reason, the answer is ``node_reporter``.
        """
        if not self.registry_dir:
            return self.node_reporter
        try:
            fd = os.open(os.path.join(self.registry_dir, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError:
            return self.node_reporter
        os.close(fd)
        return True

    # -- one step --------------------------------------------------------------
    def _node_breakdown(self, fields: Dict[str, float]) -> str:
        """``anon X, shmem Y, page cache Z, ...`` (GiB) from the ``_NODE_FIELDS`` present."""
        return ", ".join(f"{label} {fields[k] / GB:.1f}" for k, label in _NODE_FIELDS if k in fields)

    def _table_line(self, table, fields, total, avail, low, registered) -> str:
        """The node table line: availability, totals and the ``top_n`` largest processes.

        Each process is tagged with its registry tag, or ``child of`` its
        registered parent's tag; ``low`` prefixes ``LOW MEMORY``.
        """
        top = sorted(table, key=lambda r: -r[2])[: self.top_n]
        parts = []
        for p, pp, r in top:
            who = registered.get(p)
            if who is None and pp in registered:
                who = f"child of {registered[pp]}"
            label = f" [{who}]" if who else ""
            parts.append(f"pid {p}{label} {r / GB:.1f} GB {_cmdline(self.root, str(p))}")
        head = "LOW MEMORY " if low else ""
        return (f"[GF_MEM] {head}node {self.host} avail {avail / GB:.1f} of {total / GB:.1f} GB "
                f"({self._node_breakdown(fields)} GB); largest processes of uid {self.uid}: "
                + "; ".join(parts))

    def _dump_targets(self, table, registered) -> List[int]:
        """The largest registered process, and the registered parent of the largest overall."""
        targets: List[int] = []
        by_rss = sorted(table, key=lambda r: -r[2])
        reg = [p for p, _, _ in by_rss if p in registered]
        if reg:
            targets.append(reg[0])
        if by_rss:
            p0, pp0, _ = by_rss[0]
            if p0 not in registered and pp0 in registered and pp0 not in targets:
                targets.append(pp0)
        return targets

    def sample_once(self) -> List[str]:
        """Read ``/proc`` once; return (and log) the lines this sample warrants.

        In order: this process's ``[GF_MEM]`` line when due (first sample,
        heartbeat or ``low_s`` elapsed, or an RSS / availability change past
        the thresholds); the node table (LOW MEMORY: whoever claims the
        ``low_s`` wall bucket first; otherwise the node reporter every
        ``top_s``); under LOW MEMORY with a dump signal, one
        ``asking pid ... for its Python stacks`` line and signal per
        :meth:`_dump_targets` entry not yet claimed in this ``dump_gap_s``
        bucket. Also updates :attr:`pressure`.

        Returns:
            list[str]: The lines logged (empty when ``/proc/meminfo`` is
            unreadable).
        """
        lines: List[str] = []
        fields = read_meminfo_fields(self.root)
        if fields is None:
            return lines
        total, avail = fields["MemTotal"], fields["MemAvailable"]
        table = proc_table(self.root, self.uid)
        own = next((r for p, _, r in table if p == self.pid), None)
        if own is None:
            st = _read_status(self.root, str(self.pid))
            own = st[0] if st else 0.0
        kids, n_kids = _descendants_rss(table, self.pid)
        now = self.clock()
        low = avail < self.low_frac * total
        self.pressure = avail < self.press_frac * total
        due = (self._last is None or self._last_t is None
               or now - self._last_t >= (self.low_s if low else self.heartbeat_s)
               or abs(own - self._last[0]) > self.delta
               or abs(kids - self._last[1]) > self.delta
               or abs(avail - self._last[2]) > 2.0 * self.delta)
        if due:
            lines.append(
                f"[GF_MEM] {self.tag} pid {self.pid} on {self.host}: rss {own / GB:.1f} GB, "
                f"children {kids / GB:.1f} GB ({n_kids}), node avail {avail / GB:.1f} of "
                f"{total / GB:.1f} GB")
            self._last, self._last_t = (own, kids, avail), now
        registered = self._registered() if (low or self.node_reporter) else {}
        if low:
            if self._claim(f"table_{int(self.wall() // self.low_s)}"):
                lines.append(self._table_line(table, fields, total, avail, True, registered))
        elif self.node_reporter and (self._last_top_t is None or now - self._last_top_t >= self.top_s):
            lines.append(self._table_line(table, fields, total, avail, False, registered))
            self._last_top_t = now
        for ln in lines:
            self.log(ln)
        if low and self.dump_signal is not None:
            bucket = int(self.wall() // self.dump_gap_s)
            for pid in self._dump_targets(table, registered):
                if not self._claim(f"dump_{pid}_{bucket}"):
                    continue
                msg = (f"[GF_MEM] LOW MEMORY on {self.host}: asking pid {pid} "
                       f"[{registered.get(pid, '?')}] for its Python stacks (signal "
                       f"{int(self.dump_signal)}; faulthandler writes them to its stderr)")
                self.log(msg)
                lines.append(msg)
                try:
                    self.kill(pid, self.dump_signal)
                except OSError:
                    pass
        return lines

    # -- the thread -------------------------------------------------------------
    def run(self):  # pragma: no cover - timing loop; sample_once is tested
        """Sample until :meth:`stop`, every ``fast_period`` under pressure, else ``period``.

        A failing sample is swallowed (telemetry must never take the rank down).
        """
        while not self._halt.is_set():
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001 -- telemetry must never take the rank down
                pass
            self._halt.wait(self.fast_period if self.pressure else self.period)

    def stop(self):
        """Ask the loop to exit at its next wake-up (the production run never calls it)."""
        self._halt.set()


def start(tag: Optional[str] = None, *, root: str = "/proc",
          registry_dir: Optional[str] = None) -> Optional[MemWatch]:
    """Start this process's memory thread once (idempotent); ``None`` when off or off-Linux.

    Also registers :mod:`faulthandler` on ``SIGWINCH`` (all threads, to stderr)
    and this pid in the per-job registry, so a LOW MEMORY sample on the node
    can ask this process for its Python stacks.

    The thread's cadences and thresholds come from the ``GF_MEMWATCH_*``
    knobs (module docstring); the node reporter is the process whose local
    rank is 0 (or every process when no local-rank variable is set). Called
    once per process by the production driver's ``main()``.

    Args:
        tag: Label of the process in the lines, e.g. ``"w5"`` (world rank 5).
            Default: ``w`` + the MPI rank from the environment
            (``PMI_RANK`` / ``OMPI_COMM_WORLD_RANK`` / ``PMIX_RANK``), else
            ``pid`` + the pid.
        root: The ``/proc`` mount (tests point it at a fake tree).
        registry_dir: The pid registry / lock directory. Default
            :func:`default_registry_dir` (``/tmp/gf_memwatch_{SLURM_JOB_ID}``).

    Returns:
        MemWatch or None: The running thread (the same one on a repeat
        call), or ``None`` with ``GF_MEMWATCH=0`` / ``false`` / ``no`` /
        ``off``, without a readable ``/proc/meminfo``, or when the thread
        cannot start. With ``GF_MEMWATCH_DUMP`` off, or when
        :mod:`faulthandler` cannot register, no dump signal is sent.

    Example:
        .. code-block:: python

            from mpi4py import MPI
            from lisatools.globalfit import memwatch

            memwatch.start(f"w{MPI.COMM_WORLD.Get_rank()}")
    """
    global _STARTED
    if _STARTED is not None:
        return _STARTED
    if os.environ.get("GF_MEMWATCH", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if read_meminfo(root) is None:
        return None
    if tag is None:
        r = next((os.environ[k] for k in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "PMIX_RANK")
                  if k in os.environ), None)
        tag = f"w{r}" if r is not None else f"pid{os.getpid()}"
    dump = os.environ.get("GF_MEMWATCH_DUMP", "1").strip().lower() not in ("0", "false", "no", "off")
    sig = _DUMP_SIGNAL if dump else None
    if sig is not None:
        try:
            import faulthandler
            import sys

            faulthandler.register(sig, file=sys.stderr, all_threads=True, chain=False)
        except (ImportError, AttributeError, RuntimeError, ValueError, OSError):
            sig = None
    lr = _local_rank()
    w = MemWatch(
        tag, root=root,
        period=_env_float("GF_MEMWATCH_PERIOD", 30.0),
        fast_period=_env_float("GF_MEMWATCH_FAST_PERIOD", 5.0),
        press_frac=_env_float("GF_MEMWATCH_PRESS_FRAC", 0.30),
        delta_gb=_env_float("GF_MEMWATCH_DELTA_GB", 2.0),
        heartbeat_s=_env_float("GF_MEMWATCH_HEARTBEAT_S", 600.0),
        top_s=_env_float("GF_MEMWATCH_TOP_S", 600.0),
        low_frac=_env_float("GF_MEMWATCH_LOW_FRAC", 0.15),
        low_s=_env_float("GF_MEMWATCH_LOW_S", 15.0),
        dump_gap_s=_env_float("GF_MEMWATCH_DUMP_GAP_S", 120.0),
        node_reporter=(lr is None or lr == 0),
        registry_dir=registry_dir if registry_dir is not None else default_registry_dir(),
        dump_signal=sig)
    w.register()
    try:
        w.start()
    except RuntimeError:
        return None
    _STARTED = w
    return w
