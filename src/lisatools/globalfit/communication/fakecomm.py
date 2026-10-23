"""In-process stand-in for the mpi4py communicator subset the global fit uses.

``FakeWorld(size)`` runs one Python thread per rank; ``FakeComm`` implements
``Get_rank / Get_size / Get_processor_name / send / isend / recv / iprobe /
bcast / Bcast / allgather / barrier / Split / Split_type / Abort / Free``
with pickle-copy transport, so the SAME fan-out code runs on a laptop with
no MPI installed, in one python process. A test harness, not a performance
tool.

What the fake does NOT model:

- ``tag`` is ignored: there is one FIFO queue per ``(source, dest)`` pair,
  not per ``(source, dest, tag)``.
- ``Split_type`` ignores its ``split_type`` argument and always groups by
  node (the only grouping this codebase needs).
- ``send``/``isend`` are a blocking pickle-copy into an unbounded queue:
  no rendezvous handshake, no backpressure. Ordering bugs that only appear
  under real MPI's rendezvous protocol for large messages (see fanout.py's
  ``WalkerFanout.run`` isend/wait ordering, finding F1) cannot be
  reproduced here.

Not used by the production runs; it is the harness of the multi-rank tests
(``tests/test_fakecomm.py``, ``tests/test_fanout_fakecomm.py`` and the
``test_multirank_*`` smoke tests).

Example:
    .. code-block:: python

        from lisatools.globalfit.communication import FakeWorld

        world = FakeWorld(3, nodes=[0, 0, 1])

        def body(rank, comm):
            return comm.allgather(rank * 10)

        results = world.run(body)   # {0: [0, 10, 20], 1: [...], 2: [...]}
"""

from __future__ import annotations

import pickle
import queue
import threading
import time
import traceback

import numpy as np

#: mirrors of the mpi4py constants the layout code needs (only identity matters)
COMM_TYPE_SHARED = 1
UNDEFINED = -32766


class FakeAbort(RuntimeError):
    """Raised in every rank's thread once any rank aborts the world."""


def _pcopy(obj):
    """Deep copy through pickle (the transport of every fake message)."""
    return pickle.loads(pickle.dumps(obj))


class _Request:
    """Completed request returned by :meth:`FakeComm.isend` (the send already happened)."""

    def wait(self):
        return None

    def test(self):
        return True, None


class _Group:
    """Shared plumbing of one communicator: inboxes + collective slots."""

    def __init__(self, members):
        self.members = tuple(members)  # world ranks, in local-rank order
        n = len(self.members)
        self.inbox = {(d, s): queue.Queue() for d in range(n) for s in range(n)}
        self.barrier = threading.Barrier(n) if n > 0 else None
        self.slots = [None] * n


class FakeCommNull:
    """What ``Split`` returns to a rank with colour ``UNDEFINED``."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError("FakeCommNull: this rank is not a member of the communicator")


class FakeComm:
    """One rank's view of a fake communicator (mpi4py ``Comm`` subset).

    Obtained from :meth:`FakeWorld.comm` (the world communicator) or from
    :meth:`Split` / :meth:`Split_type`. Point-to-point messages go through
    one FIFO queue per ``(dest, source)`` pair; collectives use a shared
    slot exchange guarded by a :class:`threading.Barrier`, so every member
    must call them in the same order. Every blocking call raises
    :class:`FakeAbort` once any rank aborts the world.

    Args:
        world (FakeWorld): The owning world.
        group (_Group): The communicator's shared plumbing.
        world_rank (int): This rank's world rank.

    Attributes:
        COMM_TYPE_SHARED (int): Mirror of the mpi4py constant.
        UNDEFINED (int): Split colour that yields :class:`FakeCommNull`.
    """

    COMM_TYPE_SHARED = COMM_TYPE_SHARED
    UNDEFINED = UNDEFINED

    def __init__(self, world, group, world_rank):
        self._world = world
        self._group = group
        self._world_rank = int(world_rank)
        self._rank = group.members.index(self._world_rank)

    # -- topology ---------------------------------------------------------
    def Get_rank(self):
        """Return this rank's index in the communicator."""
        return self._rank

    def Get_size(self):
        """Return the number of members."""
        return len(self._group.members)

    def Get_processor_name(self):
        """Return ``"node<k>"`` for this rank's node in :attr:`FakeWorld.nodes`."""
        return self._world.node_name(self._world_rank)

    def Free(self):
        """No-op (mpi4py API compatibility)."""
        return None

    # -- point to point ---------------------------------------------------
    def send(self, obj, dest, tag=0):
        """Queue a pickle copy of ``obj`` for rank ``dest`` (``tag`` ignored)."""
        self._check_abort()
        self._group.inbox[(int(dest), self._rank)].put(_pcopy(obj))

    def isend(self, obj, dest, tag=0):
        """Same as :meth:`send`; returns an already-complete request."""
        self.send(obj, dest, tag)
        return _Request()

    def recv(self, source, tag=0):
        """Block until a message from ``source`` arrives and return it.

        Polls every 20 ms and raises :class:`FakeAbort` if the world is
        aborted meanwhile. ``tag`` is ignored.
        """
        q = self._group.inbox[(self._rank, int(source))]
        while True:
            self._check_abort()
            try:
                return q.get(timeout=0.02)
            except queue.Empty:
                continue

    def iprobe(self, source, tag=0):
        """Return ``True`` when a message from ``source`` is waiting."""
        return not self._group.inbox[(self._rank, int(source))].empty()

    # -- collectives (every member calls them in the same order) ----------
    def _wait_barrier(self):
        """Barrier wait with the world timeout; a broken barrier raises :class:`FakeAbort`."""
        self._check_abort()
        try:
            self._group.barrier.wait(timeout=self._world.timeout)
        except threading.BrokenBarrierError:
            raise FakeAbort(
                f"rank {self._world_rank}: collective broken (abort or timeout)"
            ) from None

    def _exchange(self, value):
        """Publish ``value`` in this rank's slot and return every member's slot."""
        group = self._group
        group.slots[self._rank] = value
        self._wait_barrier()
        out = list(group.slots)
        self._wait_barrier()  # nobody overwrites the slots before everyone has read
        return out

    def bcast(self, obj, root=0):
        """Return a copy of ``root``'s ``obj`` on every member (collective)."""
        out = self._exchange(_pcopy(obj) if self._rank == int(root) else None)
        return _pcopy(out[int(root)])

    def Bcast(self, buf, root=0):
        """Buffer broadcast (uppercase MPI form): fills ``buf`` IN PLACE.

        mpi4py's ``Bcast`` moves contiguous buffers without pickling, which
        is what the F-stat reference row pair (tens of MB) uses. The fake
        moves the bytes through the existing slot exchange and writes them
        into each non-root rank's own array, so a caller that allocates its
        own receive buffer -- the real usage -- takes the same shape here.

        The receive buffer must be C-CONTIGUOUS: ``reshape(-1)`` returns a
        view only then, and on a strided array it would copy, so the write
        would land in a temporary and the broadcast would vanish with no
        error at all. Refused loudly instead. (Every array the F-stat op
        broadcasts is a fresh ``np.empty``, so this never fires in
        production -- it exists so a future caller cannot fail silently.)

        What this does NOT model, on top of the module-level list:

        - the ROOT's buffer is never validated. Real mpi4py raises
          ``BufferError`` for a non-contiguous send buffer; here the root
          goes through ``tobytes()``, which happily linearizes a strided
          array. So a strided buffer is rejected by BOTH implementations but
          for different reasons and on different ranks -- receiver-side here,
          sender-side there.
        - the ``[buf, count, datatype]`` list form is accepted, but only
          ``buf`` is honoured: an explicit ``count`` or MPI datatype is
          dropped, and the whole array is sent at its own numpy dtype.
        """
        arr = buf[0] if isinstance(buf, (list, tuple)) else buf
        arr = np.asarray(arr)
        payload = self._exchange(
            arr.tobytes() if self._rank == int(root) else None)[int(root)]
        if self._rank != int(root):
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "Bcast needs a C-contiguous receive buffer; got shape "
                    f"{arr.shape} strides {arr.strides}, whose reshape(-1) "
                    "is a COPY -- the broadcast would be silently lost")
            flat = arr.reshape(-1)
            got = np.frombuffer(payload, dtype=flat.dtype)
            if got.size != flat.size:
                raise ValueError(
                    f"Bcast buffer size mismatch: root sent {got.size} "
                    f"elements, this rank's buffer holds {flat.size}")
            flat[:] = got
        return None

    def allgather(self, obj):
        """Return the list of every member's ``obj`` in rank order (collective)."""
        return [_pcopy(v) for v in self._exchange(_pcopy(obj))]

    def barrier(self):
        """Wait until every member arrives (collective)."""
        self._wait_barrier()

    Barrier = barrier

    def Split(self, color, key=0):
        """Split by ``color``, ordering members by ``(key, world rank)`` (collective).

        Returns:
            FakeComm or FakeCommNull: The new communicator, or the null comm
            for ``color == UNDEFINED``.
        """
        colors = self._exchange((color, key))
        gid = self._exchange(self._world.next_group_id() if self._rank == 0 else None)[0]
        if color == UNDEFINED:
            return FakeCommNull()
        order = sorted(
            (i for i, (c, _k) in enumerate(colors) if c == color),
            key=lambda i: (colors[i][1], self._group.members[i]),
        )
        members = tuple(self._group.members[i] for i in order)
        return FakeComm(self._world, self._world.group(gid, members), self._world_rank)

    def Split_type(self, split_type, key=0):
        """Split by node (``split_type`` ignored), like a shared-memory split."""
        return self.Split(color=self._world.node_of(self._world_rank), key=key)

    # -- failure ----------------------------------------------------------
    def Abort(self, errorcode=1):
        """Abort the world (breaks every barrier) and raise :class:`FakeAbort` here."""
        self._world.abort(errorcode, self._world_rank)
        raise FakeAbort(f"rank {self._world_rank} called Abort({errorcode})")

    def _check_abort(self):
        """Raise :class:`FakeAbort` when the world has been aborted."""
        if self._world.aborted is not None:
            code, who = self._world.aborted
            raise FakeAbort(f"rank {self._world_rank}: world aborted by rank {who} (code {code})")


class FakeWorld:
    """``size`` ranks on ``nodes`` (e.g. ``[0, 1, 0, 1, 0]``), one thread each.

    Args:
        size (int): Number of ranks.
        nodes (list[int], optional): Node id per rank (drives
            ``Get_processor_name`` and ``Split_type``). Default ``None``
            puts every rank on node 0.
        timeout (float): Seconds a collective (and :meth:`run`) may wait.
            Default ``60.0``.

    Raises:
        ValueError: When ``nodes`` does not have one entry per rank.

    Attributes:
        size (int): Number of ranks.
        nodes (list[int]): Node id per rank.
        timeout (float): Collective / run timeout in seconds.
        aborted (tuple[int, int] or None): ``(errorcode, rank)`` once a rank
            aborted, else ``None``.
    """

    def __init__(self, size, nodes=None, timeout=60.0):
        self.size = int(size)
        self.nodes = list(nodes) if nodes is not None else [0] * self.size
        if len(self.nodes) != self.size:
            raise ValueError("nodes must have one entry per rank")
        self.timeout = float(timeout)
        self.aborted = None
        self._groups = {}
        self._lock = threading.Lock()
        self._gid = 0
        self._world_group = self.group(0, tuple(range(self.size)))

    def node_of(self, rank):
        """Return the node id of world rank ``rank``."""
        return int(self.nodes[int(rank)])

    def node_name(self, rank):
        """Return ``"node<k>"`` for world rank ``rank``."""
        return f"node{self.node_of(rank)}"

    def next_group_id(self):
        """Return a fresh communicator id (thread-safe)."""
        with self._lock:
            self._gid += 1
            return self._gid

    def group(self, gid, members):
        """Return the shared plumbing of communicator ``(gid, members)``, creating it once."""
        with self._lock:
            key = (int(gid), tuple(members))
            if key not in self._groups:
                self._groups[key] = _Group(members)
            return self._groups[key]

    def comm(self, rank):
        """Return world rank ``rank``'s :class:`FakeComm` on the world communicator."""
        return FakeComm(self, self._world_group, int(rank))

    def abort(self, code, rank):
        """Mark the world aborted by ``rank`` and break every communicator's barrier."""
        self.aborted = (int(code), int(rank))
        with self._lock:
            for g in self._groups.values():
                if g.barrier is not None:
                    g.barrier.abort()

    def run(self, fn, ranks=None, timeout=None):
        """Run ``fn(rank, comm)`` on one thread per rank; return ``{rank: result}``.

        Re-raises the first NON-abort failure (the root cause) as a
        ``RuntimeError`` carrying the remote traceback text, after aborting
        the world so no other thread stays blocked.

        Args:
            fn (Callable[[int, FakeComm], object]): The per-rank body.
            ranks (list[int], optional): Ranks to run. Default ``None`` runs
                every rank.
            timeout (float, optional): Overall seconds. Default ``None`` uses
                :attr:`timeout`.

        Returns:
            dict[int, object]: ``{rank: fn's return value}``.

        Raises:
            RuntimeError: When any rank raised (lowest rank with a non-abort
                failure is reported).
            TimeoutError: When ranks are still running at the deadline and
                none failed.
        """
        ranks = list(range(self.size)) if ranks is None else list(ranks)
        results, errors = {}, {}

        def _target(r):
            try:
                results[r] = fn(r, self.comm(r))
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[r] = (exc, traceback.format_exc())
                if self.aborted is None:
                    self.abort(1, r)

        threads = [
            threading.Thread(target=_target, args=(r,), name=f"fake-rank-{r}", daemon=True)
            for r in ranks
        ]
        for t in threads:
            t.start()
        deadline = time.time() + (self.timeout if timeout is None else float(timeout))
        for t in threads:
            t.join(max(0.0, deadline - time.time()))
        alive = [t.name for t in threads if t.is_alive()]
        if alive and not errors:
            self.abort(1, -1)
            raise TimeoutError(f"FakeWorld.run: ranks still running at timeout: {alive}")
        if errors:
            root = [r for r, (e, _) in errors.items() if not isinstance(e, FakeAbort)]
            r = min(root) if root else min(errors)
            exc, tb = errors[r]
            raise RuntimeError(f"rank {r} failed:\n{tb}") from exc
        return results
